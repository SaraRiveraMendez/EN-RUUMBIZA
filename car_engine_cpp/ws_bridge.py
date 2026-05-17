# ws_bridge.py
# ================================================================
# Puente WebSocket — lanza car_inference.exe como subproceso
# y retransmite sus diagnósticos JSON a la app HTML.
#
# Uso:
#   python ws_bridge.py COM9 model.onnx
#
# La app HTML se conecta a: ws://10.43.33.139:8080
# ================================================================

import asyncio
import websockets
import subprocess
import sys
import json

HOST = "0.0.0.0"
PORT = 8080
EXE = r".\car_inference.exe"

connected_clients = set()


async def handler(websocket):
    connected_clients.add(websocket)
    print(f"[WS] Cliente conectado. Total: {len(connected_clients)}", flush=True)
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)
        print(f"[WS] Cliente desconectado. Total: {len(connected_clients)}", flush=True)


async def read_process_and_broadcast(com_port, model_path):
    cmd = [EXE, com_port, model_path]
    print(f"[Bridge] Lanzando: {' '.join(cmd)}", flush=True)

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )

    print(f"[Bridge] Proceso PID {proc.pid} iniciado", flush=True)

    while True:
        line = await proc.stdout.readline()
        if not line:
            print("[Bridge] Proceso terminó.", flush=True)
            break

        line = line.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        print(f"[exe] {line}", flush=True)

        if line.startswith("{"):
            try:
                data = json.loads(line)
                if "is_fault" in data and connected_clients:
                    await asyncio.gather(
                        *[client.send(line) for client in connected_clients],
                        return_exceptions=True,
                    )
            except json.JSONDecodeError:
                pass


async def main():
    if len(sys.argv) < 3:
        print("Uso: python ws_bridge.py <COM> <modelo.onnx>")
        sys.exit(1)

    com_port = sys.argv[1]
    model_path = sys.argv[2]

    print(f"[WS] Servidor WebSocket en ws://{HOST}:{PORT}", flush=True)
    print(f"[WS] La app HTML debe conectarse a ws://10.43.33.139:{PORT}", flush=True)

    async with websockets.serve(handler, HOST, PORT):
        await read_process_and_broadcast(com_port, model_path)


asyncio.run(main())

# EN-RUUMBIZA

Detector de fallas de motor basado en audio para Guadalahacks 2026.

## Descripción

EN-RUUMBIZA es un sistema de diagnóstico de anomalías en motores de automóvil que utiliza señales de audio. El proyecto entrena un modelo multi-etiqueta a partir de características MFCC y exporta el clasificador a ONNX para inferencia eficiente.

## Características

- Entrenamiento de un modelo de clasificación multi-etiqueta con PyTorch.
- Exportación del modelo a ONNX para desplegar en entornos de inferencia.
- Interfaz web de visualización BLE en `index.html`.
- Código C++ para extracción de características y procesamiento de audio.

## Estructura del proyecto

- `best_model/`
  - `model.onnx` — modelo ONNX listo para inferencia.
  - `cardiag_config.json` — configuración de etiquetas, umbrales y normalización.
  - `metrics.json` — métricas de evaluación.
- `car_engine_cpp/` — módulos C++ de DSP y procesamiento de audio.
- `data/` — datos de entrenamiento en formato CSV (`mfcc_features.csv`).
- `main.cpp` — prueba básica de `libsamplerate`.
- `train.py` — script principal de entrenamiento y exportación a ONNX.
- `process_all.ps1` — script PowerShell para procesar un dataset WAV y generar CSV.
- `index.html` — interfaz web para monitorizar fallas.

## Etiquetas de diagnóstico

El modelo está entrenado para detectar fallas en tres estados operacionales del motor:

### Braking State (Estado de Frenado)

- **Worn-Out Brakes** — Frenos desgastados. Las pastillas o discos de freno están muy gastados y ya no detienen el auto con eficiencia. Peligroso porque aumenta la distancia de frenado.
- **Normal Brakes** — Frenos en buen estado, funcionando correctamente.

### Idle State (Estado en Ralentí)

Motor encendido pero sin acelerar. El sistema detecta:

- **Low Oil** — Aceite bajo. El motor no tiene suficiente lubricación, lo que causa fricción excesiva entre las piezas metálicas. Si se ignora puede fundir el motor completamente.
- **Power Steering** — Fallo en la dirección asistida. El volante se vuelve muy difícil de girar, especialmente al estacionarse o a bajas velocidades.
- **Serpentine Belt** — La correa serpentina está dañada o rota. Esta correa mueve componentes vitales como el alternador, la bomba de agua y el compresor del AC. Si se rompe el auto puede sobrecalentarse y quedarse sin batería.
- **Normal Engine Idle** — Motor en ralentí funcionando correctamente, sin anomalías.

### Start-Up State (Estado de Arranque)

Al encender el auto, el sistema detecta:

- **Bad Ignition** — Fallo en el sistema de encendido. Las bujías, bobinas o el switch de encendido no están funcionando bien, causando que el motor no arranque o arranque con dificultad.
- **Dead Battery** — Batería débil o descargada. No tiene suficiente carga para arrancar el motor. Síntoma típico: el motor gira lento o no gira al girar la llave.
- **Normal Engine Start-Up** — Arranque normal, todos los sistemas respondiendo correctamente.

## Requisitos

- Python 3.9+ / 3.10+
- Paquetes Python:
  - `numpy`
  - `pandas`
  - `torch`
  - `scikit-learn`
  - `librosa`
  - `onnxruntime`

Instalación sugerida:

```bash
python -m pip install numpy pandas torch scikit-learn librosa onnxruntime
```

Para compilar `main.cpp` se requiere `libsamplerate` y un compilador C++ compatible.

## Uso

### 1. Entrenar el modelo

Ejecuta `train.py` para entrenar el clasificador, evaluar el rendimiento y generar los artefactos en `best_model/`:

```bash
python train.py
```

Se generarán:

- `best_model/model.onnx`
- `best_model/cardiag_config.json`
- `best_model/metrics.json`

### 2. Interfaz web BLE

`index.html` es una interfaz de monitorización de fallas. Ábrela en un navegador compatible y conéctala al backend BLE/WebSocket que envíe los resultados de inferencia.

## Formato de datos

El dataset `data/mfcc_features.csv` debe contener:

- una columna `label`
- columnas de características MFCC: `mfcc_0`, `mfcc_1`, ..., `mfcc_12`

El script `train.py` también puede adaptarse para cargar otros formatos, como:

- `data/features.npz`
- `data/features.jsonl`

## Notas importantes

- `train.py` realiza validación cruzada y ajuste de umbrales por clase.
- El modelo ONNX exportado incluye normalización de características.
- `process_all.ps1` es una utilidad para generar CSVs desde un dataset WAV mediante un ejecutable C++.

## Contacto

Proyecto desarrollado para Guadalahacks 2026 como herramienta de detección temprana de fallas en motores de automóvil.

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
  
- `car_engine_cpp/` — módulos C++ y Python para procesamiento de audio
  - `car_cpu/`
    - `main_inference.cpp` — código de inferencia para ECU del automóvil.
    - `model.onnx` — copia del modelo para inferencia en tiempo real.
  - `pc/`
    - `fft_features.cpp` — extractor de características FFT en C++.
    - `main_offline.cpp` — procesamiento offline de archivos WAV.
    - `MFCC.ipynb` — notebook Jupyter con análisis de características MFCC.
    - `mfcc_features.py` — script Python para extracción de características MFCC.
  - `src/`
    - `dsp_config.h` — configuración de DSP.
    - `sliding_window.cpp` — procesamiento por ventanas deslizantes.
    - `sliding_window.h` — header del procesamiento por ventanas.
  - `ws_bridge.py` — puente WebSocket para comunicación entre PC y automóvil.
  - `car_inference.exe` — ejecutable de inferencia compilado.
  - `onnxruntime.dll` — librería ONNX Runtime.
  
- `data/` — datos de entrenamiento
  - `mfcc_features.csv` — características MFCC etiquetadas para entrenamiento.
  
- `images/` — assets visuales
  - `logo.png` — logo del proyecto EN-RUUMBIZA.
  - `logo_texto.png` — logo con texto.
  
- `webapp/` — aplicaciones web
  - `enrumbiza.html` — interfaz web alternativa para monitoreo.
  
- `preprocessing.cpp` — utilidad de preprocesamiento de audio en C++.
- `main.cpp` — prueba básica de `libsamplerate`.
- `train.py` — script principal de entrenamiento y exportación a ONNX.
- `process_all.ps1` — script PowerShell para procesar dataset WAV y generar CSV.
- `index.html` — interfaz web BLE para monitorización de fallas.

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

### 2. Interfaz web BLE (`index.html`)

`index.html` es una interfaz de monitorización de fallas en tiempo real. Abre el archivo en un navegador compatible y conéctalo al backend BLE/WebSocket que envíe los resultados de inferencia.

### 3. Interfaz web alternativa (`webapp/enrumbiza.html`)

`webapp/enrumbiza.html` proporciona una alternativa de interfaz web con estilos y funcionalidades personalizadas para monitorizar fallas detectadas.

### 4. Puente WebSocket (`car_engine_cpp/ws_bridge.py`)

El script `ws_bridge.py` permite la comunicación entre el modelo de inferencia y las interfaces web mediante WebSocket:

```bash
python car_engine_cpp/ws_bridge.py
```

### 5. Procesamiento de audio en C++

#### FFT Features (Características FFT)

```bash
car_engine_cpp/pc/fft_features.cpp
```

Extrae características espectrales de archivos WAV.

#### Procesamiento offline

```bash
car_engine_cpp/pc/main_offline.cpp
```

Procesa archivos de audio sin conexión en tiempo real.

#### MFCC Features (Características MFCC)

Extrae características MFCC usando:

- `car_engine_cpp/pc/mfcc_features.py` — versión Python.
- Notebook Jupyter: `car_engine_cpp/pc/MFCC.ipynb` — análisis interactivo.

### 6. Inferencia en la ECU del automóvil

El código compilado `car_engine_cpp/car_cpu/main_inference.cpp` permite ejecutar inferencia directamente en la unidad de control del automóvil usando el modelo ONNX incluido en `car_engine_cpp/car_cpu/model.onnx`.

### 7. Procesamiento por lotes

El script `process_all.ps1` es una utilidad PowerShell para procesar un dataset completo de archivos WAV y generar CSVs etiquetados:

```powershell
.\process_all.ps1
```

## Formato de datos

El dataset `data/mfcc_features.csv` debe contener:

- una columna `label`
- columnas de características MFCC: `mfcc_0`, `mfcc_1`, ..., `mfcc_12`

El script `train.py` también puede adaptarse para cargar otros formatos, como:

- `data/features.npz`
- `data/features.jsonl`

## Arquitectura del sistema

El proyecto está diseñado con una arquitectura multi-capas:

1. **Capa de Adquisición de Audio**
   - Grabación desde micrófono (a través de librerías como `sounddevice`)
   - Lectura de archivos WAV

2. **Capa de Extracción de Características**
   - C++: `car_engine_cpp/pc/fft_features.cpp` — análisis FFT
   - Python/C++: `car_engine_cpp/pc/mfcc_features.py` — extracción MFCC
   - Ventanas deslizantes: `car_engine_cpp/src/sliding_window.cpp`

3. **Capa de Inferencia**
   - Modelo ONNX: `best_model/model.onnx` o `car_engine_cpp/car_cpu/model.onnx`
   - Inferencia en PC: `car_engine_cpp/pc/main_offline.cpp`
   - Inferencia en ECU: `car_engine_cpp/car_cpu/main_inference.cpp`

4. **Capa de Comunicación**
   - WebSocket Bridge: `car_engine_cpp/ws_bridge.py` — comunica resultados de inferencia a interfaces web

5. **Capa de Presentación**
   - Interface BLE: `index.html` — monitorización en tiempo real
   - Interface alternativa: `webapp/enrumbiza.html` — vista personalizada

## Notas importantes

- `train.py` realiza validación cruzada y ajuste de umbrales por clase.
- El modelo ONNX exportado incluye normalización de características.
- `process_all.ps1` es una utilidad para generar CSVs desde un dataset WAV mediante un ejecutable C++.
- Los archivos `.exe` y `.dll` en `car_engine_cpp/` son binarios compilados de inferencia rápida.
- El notebook `car_engine_cpp/pc/MFCC.ipynb` proporciona análisis interactivo de características.

## Contacto

Proyecto desarrollado para Guadalahacks 2026 como herramienta de detección temprana de fallas en motores de automóvil.

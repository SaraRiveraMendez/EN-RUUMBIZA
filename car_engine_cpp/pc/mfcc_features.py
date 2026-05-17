# mfcc_features.py
# ================================================================
# Lee features.csv (espectros FFT), aplica filtros de Mel + DCT
# y genera el vector de características MFCC final.
#
# Entrada : features.csv  (263682 filas × 129 columnas)
# Salida  : mfcc_features.csv (263682 filas × n_mfcc+1 columnas)
# ================================================================

import numpy as np
import pandas as pd
from scipy.fftpack import dct

# ─────────────────────────────────────────────────────────────────
# Parámetros — deben coincidir con dsp_config.h
# ─────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_FFT       = 256          # tamaño de ventana
N_FREQS     = N_FFT // 2   # 128 bins de FFT
N_MELS      = 26           # filtros de Mel (estándar para voz/audio)
N_MFCC      = 13           # coeficientes MFCC finales

# ─────────────────────────────────────────────────────────────────
# 1. Cargar y limpiar etiquetas
# ─────────────────────────────────────────────────────────────────
print("Cargando features.csv...")
df = pd.read_csv("C:/Users/rsara/OneDrive/Documents/car_engine_cpp/features.csv")

# Limpiar espacios en etiquetas → reemplazar con guión bajo
df["label"] = df["label"].str.strip().str.replace(" ", "_")

print(f"  Filas    : {len(df)}")
print(f"  Clases   : {df['label'].nunique()}")
print(f"  Etiquetas: {sorted(df['label'].unique())}\n")

# ─────────────────────────────────────────────────────────────────
# 2. Construir banco de filtros de Mel
#
# ¿Qué es un filtro de Mel?
#   El oído humano no percibe las frecuencias de forma lineal.
#   La escala Mel comprime las frecuencias altas y expande las bajas,
#   imitando la percepción auditiva. Cada filtro es una ventana
#   triangular que promedia un rango de bins de FFT.
#
#   Conversión Hz → Mel: mel = 2595 * log10(1 + hz/700)
#   Conversión Mel → Hz: hz = 700 * (10^(mel/2595) - 1)
# ─────────────────────────────────────────────────────────────────
def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def build_mel_filterbank(n_mels, n_freqs, sample_rate):
    """
    Construye una matriz de filtros de Mel.
    Retorna: matriz (n_mels × n_freqs)
    Cada fila es un filtro triangular en el espacio de frecuencias.
    """
    # Rango de frecuencias en Mel
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(sample_rate / 2)

    # n_mels + 2 puntos equiespaciados en Mel (incluye extremos)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points  = mel_to_hz(mel_points)

    # Convertir frecuencias Hz a índices de bin FFT
    # bin_k = hz * N_FFT / sample_rate
    bins = np.floor(hz_points * n_freqs * 2 / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    # Construir filtros triangulares
    filterbank = np.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        left   = bins[m - 1]
        center = bins[m]
        right  = bins[m + 1]

        # Rampa ascendente
        for k in range(left, center):
            if center != left:
                filterbank[m-1, k] = (k - left) / (center - left)

        # Rampa descendente
        for k in range(center, right):
            if right != center:
                filterbank[m-1, k] = (right - k) / (right - center)

    return filterbank

print("Construyendo banco de filtros de Mel...")
mel_filterbank = build_mel_filterbank(N_MELS, N_FREQS, SAMPLE_RATE)
print(f"  Filtros  : {N_MELS} (de 0 Hz a {SAMPLE_RATE//2} Hz)\n")

# ─────────────────────────────────────────────────────────────────
# 3. Calcular MFCC por ventana
#
# Pasos:
#   a. Tomar el espectro de magnitudes (128 bins) de la fila
#   b. Aplicar filtros de Mel → 26 energías de banda
#   c. Logaritmo → comprime el rango dinámico (como el oído)
#   d. DCT → decorrelaciona los coeficientes → 13 MFCCs
#
# ¿Por qué DCT y no otra cosa?
#   Los filtros de Mel están correlacionados entre sí (se solapan).
#   La DCT los convierte en coeficientes independientes,
#   lo que mejora el rendimiento de los clasificadores.
# ─────────────────────────────────────────────────────────────────
print("Calculando MFCCs...")

# Extraer magnitudes FFT (columnas f0..f127)
fft_cols    = [f"f{i}" for i in range(N_FREQS)]
fft_magnitudes = df[fft_cols].values   # shape: (263682, 128)

# a+b. Aplicar banco de filtros de Mel
# (263682, 128) × (128, 26)^T → (263682, 26)
mel_energies = fft_magnitudes @ mel_filterbank.T

# c. Logaritmo (evitar log(0) sumando un valor pequeño)
log_mel = np.log(mel_energies + 1e-10)

# d. DCT → quedarse con los primeros N_MFCC coeficientes
mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :N_MFCC]

print(f"  Vector de características: {N_MFCC} coeficientes MFCC por ventana\n")

# ─────────────────────────────────────────────────────────────────
# 4. Construir DataFrame de salida y guardar
# ─────────────────────────────────────────────────────────────────
mfcc_cols = [f"mfcc_{i}" for i in range(N_MFCC)]
df_out = pd.DataFrame(mfcc, columns=mfcc_cols)
df_out.insert(0, "label", df["label"].values)

output_file = "mfcc_features.csv"
df_out.to_csv(output_file, index=False)

print("==============================================")
print(f"  Archivo generado : {output_file}")
print(f"  Filas            : {len(df_out)}")
print(f"  Columnas         : {list(df_out.columns)}")
print(f"  Clases           : {sorted(df_out['label'].unique())}")
print("==============================================")
print(df_out.head(3))
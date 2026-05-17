// src/dsp_config.h
// ================================================================
// Configuración central del pipeline DSP
// Válida para PC (offline) y Arduino Uno/Nano (ATmega328, 2KB RAM)
//
// RESTRICCIÓN DE MEMORIA:
//   ATmega328 tiene 2048 bytes de RAM.
//   Ventana de 256 int16 = 512 bytes → ~25% de la RAM total.
//   Nunca superar 256 muestras en WINDOW_SIZE en este target.
// ================================================================
#pragma once

#include <stdint.h>

// --- Audio ---
#define SAMPLE_RATE       16000u   // Hz
#define SAMPLE_BITS       16       // bits por muestra (int16_t)

// --- Sliding Window ---
#define WINDOW_SIZE       256u     // muestras (256 × 2 bytes = 512 bytes de RAM)
#define HOP_SIZE          128u     // 50% overlap
#define WINDOW_FUNC_HAMMING        // activar función de ventana Hamming

// --- Tipos ---
typedef int16_t  sample_t;        // muestra cruda (ADC o WAV)
typedef float    window_t;        // muestra después de aplicar ventana (para FFT)

// --- Hamming precomputada ---
// Se genera en PC; en Arduino se embebe como tabla en PROGMEM (Flash).
// Fórmula: w[i] = 0.54 - 0.46 * cos(2π*i / (N-1))
#define HAMMING_ALPHA  0.54f
#define HAMMING_BETA   0.46f

// --- Arduino ADC ---
// El ADC del ATmega328 es de 10 bits (0–1023).
// Se mapea a int16 centrando en 0: sample = adc_raw - 512
#define ADC_CENTER      512
#define ADC_PIN         A0         // pin de entrada del micrófono

// --- Guardas de compilación ---
// En Arduino no existe <stdio.h> ni filesystem.
// Estas macros separan el código de cada target.
#ifndef ARDUINO
  #define PC_TARGET  1
#else
  #define ARDUINO_TARGET  1
#endif

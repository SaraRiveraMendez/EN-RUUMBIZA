// src/sliding_window.h
// ================================================================
// Librería de Sliding Window — interfaz compartida PC / Arduino
//
// Responsabilidad:
//   1. Mantener un buffer circular de muestras entrantes.
//   2. Detectar cuándo hay suficientes muestras para una ventana.
//   3. Aplicar la función de ventana Hamming.
//   4. Entregar el frame listo para FFT/MFCC.
//
// Diseño para ATmega328 (2KB RAM):
//   - Sin heap: todo static/stack.
//   - Sin std::vector, sin new/delete.
//   - Tabla Hamming en PROGMEM (Flash) en Arduino.
//   - Mismo .h funciona en PC con #ifdef PC_TARGET.
// ================================================================
#pragma once

#include "dsp_config.h"
#include <stdint.h>
#include <stdbool.h>

// ─────────────────────────────────────────────────────────────────
// Tabla Hamming precomputada
// En Arduino se almacena en Flash (PROGMEM) para no consumir RAM.
// En PC se almacena en RAM normal (float array).
// ─────────────────────────────────────────────────────────────────
#ifdef ARDUINO_TARGET
  #include <avr/pgmspace.h>
  extern const float hamming_table[WINDOW_SIZE] PROGMEM;
  // Leer un valor de la tabla desde Flash:
  #define HAMMING(i)  pgm_read_float(&hamming_table[(i)])
#else
  extern const float hamming_table[WINDOW_SIZE];
  #define HAMMING(i)  hamming_table[(i)]
#endif


// ─────────────────────────────────────────────────────────────────
// Estado del procesador de ventanas (sin heap)
// ─────────────────────────────────────────────────────────────────
typedef struct {
    sample_t  buffer[WINDOW_SIZE];  // buffer circular de muestras crudas
    uint16_t  write_idx;            // índice de escritura en el buffer
    uint16_t  samples_since_hop;    // muestras acumuladas desde el último hop
    bool      window_ready;         // flag: hay una ventana lista para procesar
} SlidingWindowCtx;


// ─────────────────────────────────────────────────────────────────
// API pública
// ─────────────────────────────────────────────────────────────────

/**
 * Inicializa el contexto. Llamar una vez antes de push_sample().
 */
void sw_init(SlidingWindowCtx* ctx);

/**
 * Agrega una muestra al buffer circular.
 * Retorna true si después de esta muestra hay una ventana completa lista.
 *
 * Flujo en Arduino: llamar desde la ISR del Timer o desde loop()
 * después de leer el ADC.
 */
bool sw_push_sample(SlidingWindowCtx* ctx, sample_t sample);

/**
 * Copia la ventana actual al buffer de salida y aplica Hamming.
 * output debe tener espacio para WINDOW_SIZE floats.
 *
 * IMPORTANTE: llamar solo cuando sw_push_sample() retornó true,
 * o cuando ctx->window_ready == true.
 * Después de llamar esta función, window_ready se resetea a false.
 */
void sw_get_window(SlidingWindowCtx* ctx, window_t* output);

/**
 * Normaliza la ventana al rango [-1.0, 1.0] dividiendo por 32768.
 * (El rango de int16 es [-32768, 32767])
 * Modifica output in-place.
 */
void sw_normalize(window_t* output, uint16_t size);

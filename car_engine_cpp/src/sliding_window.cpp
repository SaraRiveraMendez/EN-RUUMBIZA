// src/sliding_window.cpp
// ================================================================
// Implementación de la librería de Sliding Window
// Compatible con PC (g++) y Arduino (avr-g++ / arm-g++)
// ================================================================

#include "sliding_window.h"

#ifdef ARDUINO_TARGET
  #include <avr/pgmspace.h>
  #include <string.h>   // memset en avr-libc
#else
  #include <string.h>
  #include <math.h>
#endif

// ─────────────────────────────────────────────────────────────────
// Tabla Hamming precomputada para N = WINDOW_SIZE = 256
//
// Generada con: w[i] = 0.54 - 0.46 * cos(2*pi*i / (N-1))
// Script de generación: docs/gen_hamming.py
//
// En Arduino: PROGMEM → almacenada en Flash (32KB), no en RAM (2KB)
// En PC:      static const float → sección .rodata del binario
// ─────────────────────────────────────────────────────────────────
#ifdef ARDUINO_TARGET
const float hamming_table[WINDOW_SIZE] PROGMEM = {
#else
const float hamming_table[WINDOW_SIZE] = {
#endif
    0.080000f, 0.080140f, 0.080558f, 0.081256f, 0.082232f, 0.083487f,
    0.085018f, 0.086825f, 0.088908f, 0.091264f, 0.093893f, 0.096793f,
    0.099962f, 0.103398f, 0.107099f, 0.111063f, 0.115287f, 0.119769f,
    0.124506f, 0.129496f, 0.134734f, 0.140219f, 0.145946f, 0.151913f,
    0.158115f, 0.164549f, 0.171211f, 0.178097f, 0.185203f, 0.192524f,
    0.200056f, 0.207794f, 0.215734f, 0.223871f, 0.232200f, 0.240716f,
    0.249413f, 0.258287f, 0.267332f, 0.276542f, 0.285912f, 0.295437f,
    0.305110f, 0.314925f, 0.324878f, 0.334960f, 0.345168f, 0.355493f,
    0.365931f, 0.376474f, 0.387117f, 0.397852f, 0.408674f, 0.419575f,
    0.430550f, 0.441591f, 0.452691f, 0.463845f, 0.475045f, 0.486285f,
    0.497557f, 0.508854f, 0.520171f, 0.531500f, 0.542834f, 0.554166f,
    0.565489f, 0.576797f, 0.588083f, 0.599340f, 0.610560f, 0.621738f,
    0.632866f, 0.643938f, 0.654946f, 0.665885f, 0.676747f, 0.687527f,
    0.698217f, 0.708810f, 0.719302f, 0.729684f, 0.739951f, 0.750097f,
    0.760115f, 0.770000f, 0.779745f, 0.789345f, 0.798793f, 0.808084f,
    0.817212f, 0.826172f, 0.834958f, 0.843565f, 0.851988f, 0.860222f,
    0.868261f, 0.876100f, 0.883736f, 0.891163f, 0.898377f, 0.905373f,
    0.912148f, 0.918696f, 0.925015f, 0.931100f, 0.936947f, 0.942554f,
    0.947916f, 0.953030f, 0.957894f, 0.962504f, 0.966858f, 0.970952f,
    0.974785f, 0.978353f, 0.981656f, 0.984690f, 0.987455f, 0.989948f,
    0.992168f, 0.994113f, 0.995782f, 0.997175f, 0.998290f, 0.999128f,
    0.999686f, 0.999965f, 0.999965f, 0.999686f, 0.999128f, 0.998290f,
    0.997175f, 0.995782f, 0.994113f, 0.992168f, 0.989948f, 0.987455f,
    0.984690f, 0.981656f, 0.978353f, 0.974785f, 0.970952f, 0.966858f,
    0.962504f, 0.957894f, 0.953030f, 0.947916f, 0.942554f, 0.936947f,
    0.931100f, 0.925015f, 0.918696f, 0.912148f, 0.905373f, 0.898377f,
    0.891163f, 0.883736f, 0.876100f, 0.868261f, 0.860222f, 0.851988f,
    0.843565f, 0.834958f, 0.826172f, 0.817212f, 0.808084f, 0.798793f,
    0.789345f, 0.779745f, 0.770000f, 0.760115f, 0.750097f, 0.739951f,
    0.729684f, 0.719302f, 0.708810f, 0.698217f, 0.687527f, 0.676747f,
    0.665885f, 0.654946f, 0.643938f, 0.632866f, 0.621738f, 0.610560f,
    0.599340f, 0.588083f, 0.576797f, 0.565489f, 0.554166f, 0.542834f,
    0.531500f, 0.520171f, 0.508854f, 0.497557f, 0.486285f, 0.475045f,
    0.463845f, 0.452691f, 0.441591f, 0.430550f, 0.419575f, 0.408674f,
    0.397852f, 0.387117f, 0.376474f, 0.365931f, 0.355493f, 0.345168f,
    0.334960f, 0.324878f, 0.314925f, 0.305110f, 0.295437f, 0.285912f,
    0.276542f, 0.267332f, 0.258287f, 0.249413f, 0.240716f, 0.232200f,
    0.223871f, 0.215734f, 0.207794f, 0.200056f, 0.192524f, 0.185203f,
    0.178097f, 0.171211f, 0.164549f, 0.158115f, 0.151913f, 0.145946f,
    0.140219f, 0.134734f, 0.129496f, 0.124506f, 0.119769f, 0.115287f,
    0.111063f, 0.107099f, 0.103398f, 0.099962f, 0.096793f, 0.093893f,
    0.091264f, 0.088908f, 0.086825f, 0.085018f, 0.083487f, 0.082232f,
    0.081256f, 0.080558f, 0.080140f, 0.080000f,
};


// ─────────────────────────────────────────────────────────────────
// Implementaciones
// ─────────────────────────────────────────────────────────────────

void sw_init(SlidingWindowCtx* ctx) {
    memset(ctx->buffer, 0, sizeof(ctx->buffer));
    ctx->write_idx        = 0;
    ctx->samples_since_hop = 0;
    ctx->window_ready     = false;
}

bool sw_push_sample(SlidingWindowCtx* ctx, sample_t sample) {
    // Escribir en buffer circular
    ctx->buffer[ctx->write_idx] = sample;
    ctx->write_idx = (ctx->write_idx + 1) % WINDOW_SIZE;
    ctx->samples_since_hop++;

    // ¿Completamos un hop?
    if (ctx->samples_since_hop >= HOP_SIZE) {
        ctx->samples_since_hop = 0;
        ctx->window_ready = true;
    }

    return ctx->window_ready;
}

void sw_get_window(SlidingWindowCtx* ctx, window_t* output) {
    // Leer el buffer circular en orden cronológico y aplicar Hamming
    for (uint16_t i = 0; i < WINDOW_SIZE; i++) {
        uint16_t buf_idx = (ctx->write_idx + i) % WINDOW_SIZE;
        float sample_f   = (float)ctx->buffer[buf_idx];
        output[i]        = sample_f * HAMMING(i);
    }
    ctx->window_ready = false;
}

void sw_normalize(window_t* output, uint16_t size) {
    // int16 tiene rango [-32768, 32767] → dividir por 32768.0
    // Se aplica después de Hamming para preservar la forma de la ventana.
    const float inv_scale = 1.0f / 32768.0f;
    for (uint16_t i = 0; i < size; i++) {
        output[i] *= inv_scale;
    }
}

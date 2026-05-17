// pc/main_offline.cpp
// ================================================================
// Target PC — Pipeline offline: .wav → Sliding Window → CSV
//
// Lee archivos .wav de 16-bit PCM, los pasa por el mismo
// SlidingWindowCtx que corre en Arduino, y exporta cada ventana
// como un CSV para el compañero de MFCC/FFT.
//
// Compilar:
//   g++ -O2 -std=c++17 -I../src pc/main_offline.cpp src/sliding_window.cpp -o bin/sliding_window_pc
//
// Uso:
//   ./bin/sliding_window_pc <archivo.wav> <etiqueta>
//   ./bin/sliding_window_pc data/raw/motor_ok/rec001.wav motor_ok
// ================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
#include <samplerate.h>

#include "../src/dsp_config.h"
#include "../src/sliding_window.h"

namespace fs = std::filesystem;


// ─────────────────────────────────────────────────────────────────
// Lector WAV mínimo (PCM 16-bit, mono o estéreo)
// Sin dependencias externas — parsea el header RIFF manualmente.
// ─────────────────────────────────────────────────────────────────

#pragma pack(push, 1)
struct WavHeader {
    char     riff[4];        // "RIFF"
    uint32_t chunk_size;
    char     wave[4];        // "WAVE"
    char     fmt[4];         // "fmt "
    uint32_t fmt_size;       // 16 para PCM
    uint16_t audio_format;   // 1 = PCM
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char     data[4];        // "data"
    uint32_t data_size;
};
#pragma pack(pop)

struct WavFile {
    FILE*    fp;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint16_t bits_per_sample;
    uint32_t num_samples;    // total de frames (muestras mono)
};

/**
 * Abre un archivo WAV y valida su cabecera.
 * Retorna false si el formato no es compatible.
 */
bool wav_open(const char* path, WavFile* wav) {
    wav->fp = fopen(path, "rb");
    if (!wav->fp) {
        fprintf(stderr, "[ERROR] No se puede abrir: %s\n", path);
        return false;
    }

    WavHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, wav->fp) != 1) {
        fprintf(stderr, "[ERROR] No se puede leer el header WAV.\n");
        fclose(wav->fp);
        return false;
    }

    if (strncmp(hdr.riff, "RIFF", 4) != 0 || strncmp(hdr.wave, "WAVE", 4) != 0) {
        fprintf(stderr, "[ERROR] No es un archivo WAV válido.\n");
        fclose(wav->fp);
        return false;
    }

    if (hdr.audio_format != 1) {
        fprintf(stderr, "[ERROR] Solo se soporta PCM (audio_format=1). Encontrado: %d\n", hdr.audio_format);
        fclose(wav->fp);
        return false;
    }

    if (hdr.bits_per_sample != 16) {
        fprintf(stderr, "[ERROR] Solo se soporta 16-bit. Encontrado: %d-bit\n", hdr.bits_per_sample);
        fclose(wav->fp);
        return false;
    }

    // if ((uint32_t)hdr.sample_rate != SAMPLE_RATE) {
    //     fprintf(stderr, "[ERROR] Sample rate incorrecto: esperado %u Hz, encontrado %u Hz.\n",
    //             SAMPLE_RATE, hdr.sample_rate);
    //     fclose(wav->fp);
    //     return false;
    // }

    wav->num_channels    = hdr.num_channels;
    wav->sample_rate     = hdr.sample_rate;
    wav->bits_per_sample = hdr.bits_per_sample;
    wav->num_samples     = hdr.data_size / (hdr.num_channels * 2);

    printf("  Canales      : %u\n",   wav->num_channels);
    printf("  Sample rate  : %u Hz\n", wav->sample_rate);
    printf("  Total frames : %u\n",   wav->num_samples);
    printf("  Duración     : %.2f s\n", (float)wav->num_samples / wav->sample_rate);

    return true;
}

/**
 * Lee el siguiente frame del WAV.
 * Si es estéreo, promedia L+R a mono.
 * Retorna false al llegar al final del archivo.
 */
bool wav_read_sample(WavFile* wav, sample_t* out) {
    int16_t raw[2] = {0, 0};
    size_t  n = fread(raw, sizeof(int16_t), wav->num_channels, wav->fp);
    if (n < wav->num_channels) return false;

    if (wav->num_channels == 1) {
        *out = raw[0];
    } else {
        // Promedio L+R sin overflow: dividir antes de sumar
        *out = (int16_t)((int32_t)raw[0] / 2 + (int32_t)raw[1] / 2);
    }
    return true;
}

void wav_close(WavFile* wav) {
    if (wav->fp) fclose(wav->fp);
}

std::vector<sample_t> resample_audio(
    const std::vector<sample_t>& input,
    int input_rate,
    int output_rate)
{
    double ratio = (double)output_rate / input_rate;

    int output_frames = (int)(input.size() * ratio) + 1;

    // int16 -> float
    std::vector<float> input_float(input.size());

    for (size_t i = 0; i < input.size(); i++) {
        input_float[i] = input[i] / 32768.0f;
    }

    std::vector<float> output_float(output_frames);

    SRC_DATA data{};
    data.data_in = input_float.data();
    data.input_frames = (long)input_float.size();

    data.data_out = output_float.data();
    data.output_frames = output_frames;

    data.src_ratio = ratio;
    data.end_of_input = 1;

    int err = src_simple(&data, SRC_SINC_FASTEST, 1);

    if (err) {
        fprintf(stderr, "[ERROR] libsamplerate: %s\n", src_strerror(err));
        return {};
    }

    // float -> int16
    std::vector<sample_t> output(data.output_frames_gen);

    for (long i = 0; i < data.output_frames_gen; i++) {

        float s = output_float[i];

        if (s > 1.0f)  s = 1.0f;
        if (s < -1.0f) s = -1.0f;

        output[i] = (sample_t)(s * 32767.0f);
    }

    return output;
}


// ─────────────────────────────────────────────────────────────────
// Exportar ventana a CSV
// ─────────────────────────────────────────────────────────────────

/**
 * Guarda una ventana como CSV.
 * Formato: header (s0,s1,...,s255) + 1 fila de valores float.
 * Nombre: {label}_{basename}_w{NNNN}.csv
 */
bool save_window_csv(
    const window_t* window,
    const char*     output_dir,
    const char*     label,
    const char*     basename,
    uint32_t        window_idx)
{
    char filepath[512];
    snprintf(filepath, sizeof(filepath),
             "%s/%s_%s_w%04u.csv",
             output_dir, label, basename, window_idx);

    FILE* f = fopen(filepath, "w");
    if (!f) {
        fprintf(stderr, "[ERROR] No se puede crear: %s\n", filepath);
        return false;
    }

    // Header
    for (uint16_t i = 0; i < WINDOW_SIZE; i++) {
        fprintf(f, "s%u", i);
        if (i < WINDOW_SIZE - 1) fputc(',', f);
    }
    fputc('\n', f);

    // Datos
    for (uint16_t i = 0; i < WINDOW_SIZE; i++) {
        fprintf(f, "%.6f", window[i]);
        if (i < WINDOW_SIZE - 1) fputc(',', f);
    }
    fputc('\n', f);

    fclose(f);
    return true;
}


// ─────────────────────────────────────────────────────────────────
// Pipeline principal
// ─────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {

    if (argc < 3) {
        fprintf(stderr,
            "Uso: %s <archivo.wav> <etiqueta>\n"
            "Ejemplo: %s data/raw/motor_ok/rec001.wav motor_ok\n",
            argv[0], argv[0]);
        return 1;
    }

    const char* wav_path = argv[1];
    const char* label    = argv[2];

    // Extraer nombre base del archivo
    fs::path p(wav_path);
    std::string basename = p.stem().string();

    // Carpeta de salida
    char output_dir[256];
    snprintf(output_dir, sizeof(output_dir),
             "data/windows/%s", label);

    fs::create_directories(output_dir);

    printf("\n==============================================\n");
    printf("  Archivo  : %s\n", wav_path);
    printf("  Etiqueta : %s\n", label);
    printf("  Salida   : %s/\n", output_dir);
    printf("==============================================\n");

    // Abrir WAV
    WavFile wav;

    if (!wav_open(wav_path, &wav)) {
        return 1;
    }

    // Sliding Window
    SlidingWindowCtx ctx;
    sw_init(&ctx);

    window_t frame[WINDOW_SIZE];

    uint32_t window_count = 0;
    uint32_t sample_count = 0;

    printf("\nCargando audio...\n");

    // Leer TODO el audio
    std::vector<sample_t> audio;

    sample_t sample;

    while (wav_read_sample(&wav, &sample)) {
        audio.push_back(sample);
    }

    wav_close(&wav);

    printf("  Samples cargados: %zu\n", audio.size());

    // Remuestrear si hace falta
    if (wav.sample_rate != SAMPLE_RATE) {

        printf("  Remuestreando: %u Hz -> %u Hz\n",
               wav.sample_rate,
               SAMPLE_RATE);

        audio = resample_audio(
            audio,
            wav.sample_rate,
            SAMPLE_RATE);

        if (audio.empty()) {
            return 1;
        }
    }

    printf("\nProcesando...\n");

    for (sample_t sample : audio) {

        sample_count++;

        bool ready = sw_push_sample(&ctx, sample);

        if (ready) {

            sw_get_window(&ctx, frame);

            sw_normalize(frame, WINDOW_SIZE);

            save_window_csv(
                frame,
                output_dir,
                label,
                basename.c_str(),
                window_count);

            window_count++;

            if (window_count % 100 == 0) {
                printf("  Ventanas generadas: %u\r",
                       window_count);
            }
        }
    }

    printf("\n\n==============================================\n");
    printf("  RESUMEN\n");
    printf("==============================================\n");
    printf("  Muestras procesadas : %u\n", sample_count);
    printf("  Ventanas generadas  : %u\n", window_count);
    printf("  CSV guardados en    : %s/\n", output_dir);

    printf("  Config: WINDOW=%u, HOP=%u, SR=%u\n",
           WINDOW_SIZE,
           HOP_SIZE,
           SAMPLE_RATE);

    return 0;
}
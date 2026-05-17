// pc/fft_features.cpp
// ================================================================
// Lee los CSVs de sliding window, aplica FFT y genera el espectro
// de magnitudes listo para que el pipeline de MFCC lo consuma.
//
// Flujo:
//   CSV (256 muestras) → FFT → magnitudes (128 bins) → CSV de salida
//
// El CSV de salida tiene 128 columnas (f0..f127) + columna "label".
// Una fila por ventana. Tu compañero de MFCC lee ese archivo.
//
// Compilar:
//   g++ -O2 -std=c++17 pc/fft_features.cpp -lfftw3 -o fft_features.exe
//
// Uso — una carpeta de etiqueta:
//   .\fft_features.exe data\windows\worn_out_brakes worn_out_brakes
//
// Uso — todo el dataset:
//   .\fft_all.ps1
// ================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <fftw3.h>


namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────
// Constantes — deben coincidir con dsp_config.h
// ─────────────────────────────────────────────────────────────────
static const int    N_SAMPLES  = 256;        // muestras por ventana
static const int    N_FREQS    = N_SAMPLES / 2; // bins útiles de FFT (0..127)
static const double SAMPLE_RATE = 16000.0;

// ─────────────────────────────────────────────────────────────────
// Leer un CSV de ventana
// Retorna vector de N_SAMPLES floats, o vacío si hay error.
// ─────────────────────────────────────────────────────────────────
std::vector<double> read_window_csv(const fs::path& path) {
    FILE* f = fopen(path.string().c_str(), "r");
    if (!f) {
        fprintf(stderr, "[ERROR] No se puede abrir: %s\n", path.string().c_str());
        return {};
    }

    // Saltar header (primera línea: s0,s1,...,s255)
    char line[8192];
    if (!fgets(line, sizeof(line), f)) { fclose(f); return {}; }

    // Leer fila de datos
    if (!fgets(line, sizeof(line), f)) { fclose(f); return {}; }
    fclose(f);

    // Parsear valores separados por coma
    std::vector<double> samples;
    samples.reserve(N_SAMPLES);

    char* token = strtok(line, ",\n\r");
    while (token && (int)samples.size() < N_SAMPLES) {
        samples.push_back(atof(token));
        token = strtok(nullptr, ",\n\r");
    }

    if ((int)samples.size() != N_SAMPLES) {
        fprintf(stderr, "[WARN] CSV incompleto (%zu muestras): %s\n",
                samples.size(), path.filename().string().c_str());
        return {};
    }

    return samples;
}

// ─────────────────────────────────────────────────────────────────
// Aplicar FFT y calcular magnitudes
//
// Entrada:  N_SAMPLES valores de tiempo (ya con Hamming aplicada)
// Salida:   N_FREQS magnitudes normalizadas
//
// ¿Por qué solo N/2 bins?
//   La FFT de una señal real produce N coeficientes complejos,
//   pero la segunda mitad es espejo de la primera (conjugado).
//   Solo los primeros N/2 bins tienen información única.
//   El bin k corresponde a la frecuencia: k * SAMPLE_RATE / N
//   Bin 0   → 0 Hz (DC)
//   Bin 127 → 127 * 16000 / 256 = 7937.5 Hz
// ─────────────────────────────────────────────────────────────────
std::vector<double> compute_fft_magnitudes(const std::vector<double>& samples) {
    // FFTW necesita buffers alineados
    double*       in  = fftw_alloc_real(N_SAMPLES);
    fftw_complex* out = fftw_alloc_complex(N_FREQS + 1);

    // Copiar muestras al buffer de entrada
    for (int i = 0; i < N_SAMPLES; i++)
        in[i] = samples[i];

    // Crear plan y ejecutar FFT (r2c = real to complex)
    fftw_plan plan = fftw_plan_dft_r2c_1d(N_SAMPLES, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Calcular magnitudes normalizadas
    // |X[k]| = sqrt(real² + imag²) / N
    std::vector<double> magnitudes(N_FREQS);
    for (int k = 0; k < N_FREQS; k++) {
        double real = out[k][0];
        double imag = out[k][1];
        magnitudes[k] = sqrt(real * real + imag * imag) / N_SAMPLES;
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return magnitudes;
}

// ─────────────────────────────────────────────────────────────────
// Procesar una carpeta de CSVs y escribir archivo de salida
//
// Formato del CSV de salida:
//   label, f0, f1, ..., f127
//   worn_out_brakes, 0.001234, 0.002345, ...
//
// Una fila = una ventana = un vector de características listo para MFCC
// ─────────────────────────────────────────────────────────────────
bool process_label_folder(
    const fs::path& input_dir,
    const char*     label,
    const char*     output_file,
    bool            write_header)
{
    // Recopilar todos los CSVs de la carpeta
    std::vector<fs::path> csv_files;
    for (auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.path().extension() == ".csv")
            csv_files.push_back(entry.path());
    }

    if (csv_files.empty()) {
        fprintf(stderr, "[WARN] No hay CSVs en: %s\n", input_dir.string().c_str());
        return false;
    }

    // Ordenar para reproducibilidad
    std::sort(csv_files.begin(), csv_files.end());

    // Abrir archivo de salida (append si ya existe)
    FILE* out = fopen(output_file, write_header ? "w" : "a");
    if (!out) {
        fprintf(stderr, "[ERROR] No se puede crear: %s\n", output_file);
        return false;
    }

    // Escribir header solo una vez (primera etiqueta)
    if (write_header) {
        fprintf(out, "label");
        for (int k = 0; k < N_FREQS; k++)
            fprintf(out, ",f%d", k);
        fprintf(out, "\n");
    }

    int processed = 0;
    int skipped   = 0;

    for (auto& csv_path : csv_files) {
        // 1. Leer ventana
        std::vector<double> samples = read_window_csv(csv_path);
        if (samples.empty()) { skipped++; continue; }

        // 2. Aplicar FFT
        std::vector<double> magnitudes = compute_fft_magnitudes(samples);

        // 3. Escribir fila: label, f0, f1, ..., f127
        fprintf(out, "%s", label);
        for (int k = 0; k < N_FREQS; k++)
            fprintf(out, ",%.8f", magnitudes[k]);
        fprintf(out, "\n");

        processed++;
    }

    fclose(out);

    printf("  %-25s → %d ventanas procesadas", label, processed);
    if (skipped > 0) printf(" (%d saltadas)", skipped);
    printf("\n");

    return true;
}

// ─────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr,
            "Uso: %s <carpeta_windows> <archivo_salida.csv>\n"
            "Ejemplo:\n"
            "  %s data\\windows features.csv\n",
            argv[0], argv[0]);
        return 1;
    }

    fs::path windows_dir = argv[1];
    const char* output   = argv[2];

    if (!fs::exists(windows_dir)) {
        fprintf(stderr, "[ERROR] No existe: %s\n", argv[1]);
        return 1;
    }

    printf("\n==============================================\n");
    printf("  FFT Feature Extractor\n");
    printf("  Entrada : %s\n", argv[1]);
    printf("  Salida  : %s\n", output);
    printf("  Config  : N=%d, SR=%.0f, bins=%d\n",
           N_SAMPLES, SAMPLE_RATE, N_FREQS);
    printf("  Rango   : 0 Hz – %.1f Hz (%.2f Hz/bin)\n",
           SAMPLE_RATE / 2.0, SAMPLE_RATE / N_SAMPLES);
    printf("==============================================\n\n");

    bool first = true;
    int  total_labels = 0;

    // Recorrer cada subcarpeta de data/windows/ (una por etiqueta)
    for (auto& entry : fs::directory_iterator(windows_dir)) {
        if (!entry.is_directory()) continue;

        std::string label = entry.path().filename().string();
        bool ok = process_label_folder(entry.path(), label.c_str(), output, first);

        if (ok) { first = false; total_labels++; }
    }

    printf("\n==============================================\n");
    printf("  Etiquetas procesadas : %d\n", total_labels);
    printf("  Archivo generado     : %s\n", output);
    printf("==============================================\n\n");

    return 0;
}
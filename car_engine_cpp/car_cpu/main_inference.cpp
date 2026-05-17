// car_cpu/main_inference.cpp
// ================================================================
// CPU del coche — Pipeline de inferencia en tiempo real
//
// Flujo:
//   Arduino (Serial) → ventana 256 muestras
//   → FFT (128 bins)
//   → Filtros Mel (26 bandas)
//   → MFCC (13 coeficientes)
//   → ONNX Runtime → clasificación
//
// Compilar (desde car_engine_cpp/):
//   g++ -O2 -std=c++17 car_cpu/main_inference.cpp -lfftw3 ^
//       -I "C:\onnxruntime\include" ^
//       -L "C:\onnxruntime\lib" -lonnxruntime ^
//       -o car_inference.exe
//
// Antes de correr, copiar onnxruntime.dll al mismo directorio:
//   copy C:\onnxruntime\lib\onnxruntime.dll .
//
// Uso:
//   .\car_inference.exe COM9 modelo.onnx
// ================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <string>
#include <vector>
#include <array>
#include <algorithm>

// FFT
#include <fftw3.h>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <onnxruntime_cxx_api.h>

// ─────────────────────────────────────────────────────────────────
// Constantes — deben coincidir con dsp_config.h y mfcc_features.py
// ─────────────────────────────────────────────────────────────────
static const int    N_SAMPLES   = 256;
static const int    N_FREQS     = N_SAMPLES / 2;   // 128 bins FFT
static const int    N_MELS      = 26;
static const int    N_MFCC      = 13;
static const double SAMPLE_RATE = 16000.0;

// Etiquetas en el mismo orden que el entrenamiento
static const char* LABELS[] = {
    "bad_ignition",
    "dead_battery",
    "low_oil",
    "normal_brakes",
    "normal_engine_idle",
    "normal_engine_startup",
    "power_steering",
    "serpentine_belt",
    "worn_out_brakes"
};
static const int N_CLASSES = 9;


// ─────────────────────────────────────────────────────────────────
// Banco de filtros de Mel (igual que mfcc_features.py)
// ─────────────────────────────────────────────────────────────────

static double mel_filterbank[N_MELS][N_FREQS];
static bool   filterbank_initialized = false;

double hz_to_mel(double hz) {
    return 2595.0 * log10(1.0 + hz / 700.0);
}

double mel_to_hz(double mel) {
    return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
}

void init_mel_filterbank() {
    if (filterbank_initialized) return;

    memset(mel_filterbank, 0, sizeof(mel_filterbank));

    double mel_min = hz_to_mel(0.0);
    double mel_max = hz_to_mel(SAMPLE_RATE / 2.0);

    // N_MELS + 2 puntos equiespaciados en Mel
    double mel_points[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++)
        mel_points[i] = mel_min + i * (mel_max - mel_min) / (N_MELS + 1);

    // Convertir a índices de bin
    int bins[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++) {
        double hz = mel_to_hz(mel_points[i]);
        int bin   = (int)floor(hz * N_FREQS * 2.0 / SAMPLE_RATE);
        bins[i]   = std::max(0, std::min(bin, N_FREQS - 1));
    }

    // Filtros triangulares
    for (int m = 1; m <= N_MELS; m++) {
        int left   = bins[m - 1];
        int center = bins[m];
        int right  = bins[m + 1];

        for (int k = left; k < center; k++)
            if (center != left)
                mel_filterbank[m-1][k] = (double)(k - left) / (center - left);

        for (int k = center; k < right; k++)
            if (right != center)
                mel_filterbank[m-1][k] = (double)(right - k) / (right - center);
    }

    filterbank_initialized = true;
}


// ─────────────────────────────────────────────────────────────────
// FFT con FFTW3
// ─────────────────────────────────────────────────────────────────

std::vector<double> compute_fft_magnitudes(const std::vector<double>& samples) {
    double*       in  = fftw_alloc_real(N_SAMPLES);
    fftw_complex* out = fftw_alloc_complex(N_FREQS + 1);

    for (int i = 0; i < N_SAMPLES; i++)
        in[i] = samples[i];

    fftw_plan plan = fftw_plan_dft_r2c_1d(N_SAMPLES, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);

    std::vector<double> magnitudes(N_FREQS);
    for (int k = 0; k < N_FREQS; k++) {
        double re = out[k][0];
        double im = out[k][1];
        magnitudes[k] = sqrt(re * re + im * im) / N_SAMPLES;
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return magnitudes;
}


// ─────────────────────────────────────────────────────────────────
// MFCC
// ─────────────────────────────────────────────────────────────────

std::vector<float> compute_mfcc(const std::vector<double>& fft_magnitudes) {
    // 1. Aplicar filtros de Mel
    double mel_energies[N_MELS] = {0};
    for (int m = 0; m < N_MELS; m++)
        for (int k = 0; k < N_FREQS; k++)
            mel_energies[m] += fft_magnitudes[k] * mel_filterbank[m][k];

    // 2. Logaritmo
    double log_mel[N_MELS];
    for (int m = 0; m < N_MELS; m++)
        log_mel[m] = log(mel_energies[m] + 1e-10);

    // 3. DCT-II (normalizada) → primeros N_MFCC coeficientes
    // DCT-II: X[n] = sum(x[m] * cos(pi*n*(2m+1)/(2*N_MELS)))
    std::vector<float> mfcc(N_MFCC);
    double scale = sqrt(2.0 / N_MELS);
    for (int n = 0; n < N_MFCC; n++) {
        double sum = 0.0;
        for (int m = 0; m < N_MELS; m++)
            sum += log_mel[m] * cos(M_PI * n * (2*m + 1) / (2.0 * N_MELS));
        mfcc[n] = (float)(scale * sum);
    }

    return mfcc;
}


// ─────────────────────────────────────────────────────────────────
// Comunicación Serial con Arduino (Windows API)
// ─────────────────────────────────────────────────────────────────

HANDLE serial_open(const char* port, int baudrate) {
    // En Windows el puerto debe ser \\.\COM9 para COM9+
    char full_port[32];
    snprintf(full_port, sizeof(full_port), "\\\\.\\%s", port);

    HANDLE h = CreateFileA(full_port, GENERIC_READ | GENERIC_WRITE,
                           0, NULL, OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[ERROR] No se puede abrir %s\n", port);
        return INVALID_HANDLE_VALUE;
    }

    DCB dcb = {0};
    dcb.DCBlength = sizeof(dcb);
    GetCommState(h, &dcb);
    dcb.BaudRate = baudrate;
    dcb.ByteSize = 8;
    dcb.StopBits = ONESTOPBIT;
    dcb.Parity   = NOPARITY;
    SetCommState(h, &dcb);

    COMMTIMEOUTS timeouts = {0};
    timeouts.ReadIntervalTimeout         = 50;
    timeouts.ReadTotalTimeoutConstant    = 500;
    timeouts.ReadTotalTimeoutMultiplier  = 10;
    SetCommTimeouts(h, &timeouts);

    return h;
}

// Lee una línea del Serial (terminada en '\n')
std::string serial_read_line(HANDLE h) {
    std::string line;
    char c;
    DWORD read;
    while (ReadFile(h, &c, 1, &read, NULL) && read > 0) {
        if (c == '\n') break;
        if (c != '\r') line += c;
    }
    return line;
}


// ─────────────────────────────────────────────────────────────────
// Parsear línea del Arduino
//
// El Arduino envía: W0001:0.0012,0.0034,...,-0.0056
// Esta función extrae los 256 valores float.
// ─────────────────────────────────────────────────────────────────
std::vector<double> parse_arduino_window(const std::string& line) {
    std::vector<double> samples;

    // Buscar el ':' que separa el índice de los datos
    size_t colon = line.find(':');
    if (colon == std::string::npos) return samples;

    std::string data = line.substr(colon + 1);
    samples.reserve(N_SAMPLES);

    char* buf   = &data[0];
    char* token = strtok(buf, ",");
    while (token && (int)samples.size() < N_SAMPLES) {
        samples.push_back(atof(token));
        token = strtok(nullptr, ",");
    }

    return samples;
}


// ─────────────────────────────────────────────────────────────────
// Inferencia ONNX
// ─────────────────────────────────────────────────────────────────

struct OnnxModel {
    Ort::Env            env;
    Ort::Session        session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::string         input_name;
    std::string         output_name;

    OnnxModel(const char* model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "car_inference"),
          session(
    env,
    std::wstring(model_path, model_path + strlen(model_path)).c_str(),
    Ort::SessionOptions{})
    {
        // Obtener nombres de input/output
        char* in  = session.GetInputNameAllocated(0, allocator).release();
        char* out = session.GetOutputNameAllocated(0, allocator).release();
        input_name  = in;
        output_name = out;
    }

    // Retorna el índice de la clase predicha
    int predict(const std::vector<float>& mfcc_vector) {
        // Shape: [1, N_MFCC]
        std::array<int64_t, 2> shape = {1, N_MFCC};

        Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            const_cast<float*>(mfcc_vector.data()),
            mfcc_vector.size(),
            shape.data(), shape.size());

        const char* in_name  = input_name.c_str();
        const char* out_name = output_name.c_str();

        auto output = session.Run(
            Ort::RunOptions{nullptr},
            &in_name,  &input_tensor, 1,
            &out_name, 1);

        // Obtener probabilidades y encontrar la clase con mayor score
        float* scores = output[0].GetTensorMutableData<float>();
        int    best   = 0;
        for (int i = 1; i < N_CLASSES; i++)
            if (scores[i] > scores[best]) best = i;

        return best;
    }
};


// ─────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr,
            "Uso: %s <puerto_COM> <modelo.onnx>\n"
            "Ejemplo: %s COM9 modelo.onnx\n",
            argv[0], argv[0]);
        return 1;
    }

    const char* port       = argv[1];
    const char* model_path = argv[2];

    printf("\n==============================================\n");
    printf("  Car Diagnostic — Inferencia en tiempo real\n");
    printf("  Puerto : %s\n", port);
    printf("  Modelo : %s\n", model_path);
    printf("==============================================\n\n");

    // Inicializar banco de filtros Mel
    init_mel_filterbank();
    printf("[OK] Banco de filtros Mel inicializado\n");

    // Cargar modelo ONNX
    printf("[..] Cargando modelo ONNX...\n");
    OnnxModel model(model_path);
    printf("[OK] Modelo cargado\n");

    // Abrir puerto Serial
    printf("[..] Conectando a %s...\n", port);
    HANDLE serial = serial_open(port, 115200);
    if (serial == INVALID_HANDLE_VALUE) return 1;
    printf("[OK] Conectado. Esperando ventanas del Arduino...\n\n");

    printf("%-10s %-35s %s\n", "Ventana", "Diagnóstico", "Confianza");
    printf("%s\n", std::string(55, '-').c_str());

    uint32_t window_count = 0;

    while (true) {
        // 1. Leer línea del Arduino
        std::string line = serial_read_line(serial);
        if (line.empty() || line[0] != 'W') continue;

        // 2. Parsear ventana
        std::vector<double> samples = parse_arduino_window(line);
        if ((int)samples.size() != N_SAMPLES) continue;

        // 3. FFT
        std::vector<double> magnitudes = compute_fft_magnitudes(samples);

        // 4. MFCC
        std::vector<float> mfcc = compute_mfcc(magnitudes);

        // 5. Inferencia
        int predicted = model.predict(mfcc);

        printf("%-10u %-35s\n",
               window_count,
               LABELS[predicted]);

        window_count++;
    }

    CloseHandle(serial);
    return 0;
}
// car_cpu/main_inference.cpp
// ================================================================
// CPU del coche — Pipeline de inferencia en tiempo real
// Con voto por mayoría y alerta de dashboard
// ================================================================

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <onnxruntime_cxx_api.h>
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
#include <deque>
#include <fftw3.h>

static const int    N_SAMPLES   = 256;
static const int    N_FREQS     = N_SAMPLES / 2;
static const int    N_MELS      = 26;
static const int    N_MFCC      = 13;
static const double SAMPLE_RATE = 16000.0;
static const int    VOTE_WINDOW = 15;

static const char* NORMAL_CLASSES[] = {
    "normal_engine_idle", "normal_brakes", "normal_engine_startup"
};
static const int N_NORMAL = 3;

static const char* LABELS[] = {
    "bad_ignition", "dead_battery", "low_oil", "normal_brakes",
    "normal_engine_idle", "normal_engine_startup", "power_steering",
    "serpentine_belt", "worn_out_brakes"
};
static const int N_CLASSES = 9;

static const char* ALERT_MESSAGES[] = {
    "FALLA DE ENCENDIDO\nRevise el sistema de ignicion.",
    "BATERIA DESCARGADA\nRevise la bateria o el alternador.",
    "NIVEL DE ACEITE BAJO\nRevise el nivel de aceite del motor.",
    "", "",  "",
    "FALLA EN DIRECCION\nRevise el sistema de direccion hidraulica.",
    "FALLA EN CORREA\nRevise la correa serpentina.",
    "FRENOS DESGASTADOS\nRevise las balatas y discos de freno."
};

// ── Mel filterbank ───────────────────────────────────────────────
static double mel_filterbank[N_MELS][N_FREQS];
static bool   filterbank_initialized = false;

double hz_to_mel(double hz) { return 2595.0 * log10(1.0 + hz / 700.0); }
double mel_to_hz(double mel) { return 700.0 * (pow(10.0, mel / 2595.0) - 1.0); }

void init_mel_filterbank() {
    if (filterbank_initialized) return;
    memset(mel_filterbank, 0, sizeof(mel_filterbank));
    double mel_min = hz_to_mel(0.0), mel_max = hz_to_mel(SAMPLE_RATE / 2.0);
    double mel_points[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++)
        mel_points[i] = mel_min + i * (mel_max - mel_min) / (N_MELS + 1);
    int bins[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++) {
        int bin = (int)floor(mel_to_hz(mel_points[i]) * N_FREQS * 2.0 / SAMPLE_RATE);
        bins[i] = std::max(0, std::min(bin, N_FREQS - 1));
    }
    for (int m = 1; m <= N_MELS; m++) {
        int left = bins[m-1], center = bins[m], right = bins[m+1];
        for (int k = left;   k < center; k++) if (center != left)  mel_filterbank[m-1][k] = (double)(k-left)/(center-left);
        for (int k = center; k < right;  k++) if (right  != center) mel_filterbank[m-1][k] = (double)(right-k)/(right-center);
    }
    filterbank_initialized = true;
}

// ── FFT ──────────────────────────────────────────────────────────
std::vector<double> compute_fft_magnitudes(const std::vector<double>& samples) {
    double* in = fftw_alloc_real(N_SAMPLES);
    fftw_complex* out = fftw_alloc_complex(N_FREQS + 1);
    for (int i = 0; i < N_SAMPLES; i++) in[i] = samples[i];
    fftw_plan plan = fftw_plan_dft_r2c_1d(N_SAMPLES, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);
    std::vector<double> mag(N_FREQS);
    for (int k = 0; k < N_FREQS; k++) mag[k] = sqrt(out[k][0]*out[k][0] + out[k][1]*out[k][1]) / N_SAMPLES;
    fftw_destroy_plan(plan); fftw_free(in); fftw_free(out);
    return mag;
}

// ── MFCC ─────────────────────────────────────────────────────────
std::vector<float> compute_mfcc(const std::vector<double>& mag) {
    double mel[N_MELS] = {0};
    for (int m = 0; m < N_MELS; m++)
        for (int k = 0; k < N_FREQS; k++)
            mel[m] += mag[k] * mel_filterbank[m][k];
    double log_mel[N_MELS];
    for (int m = 0; m < N_MELS; m++) log_mel[m] = log(mel[m] + 1e-10);
    std::vector<float> mfcc(N_MFCC);
    double scale = sqrt(2.0 / N_MELS);
    for (int n = 0; n < N_MFCC; n++) {
        double sum = 0.0;
        for (int m = 0; m < N_MELS; m++)
            sum += log_mel[m] * cos(M_PI * n * (2*m+1) / (2.0*N_MELS));
        mfcc[n] = (float)(scale * sum);
    }
    return mfcc;
}

// ── Voto por mayoría ─────────────────────────────────────────────
struct MajorityVoter {
    std::deque<int> history;
    int vote(int pred) {
        history.push_back(pred);
        if ((int)history.size() > VOTE_WINDOW) history.pop_front();
        int counts[N_CLASSES] = {0};
        for (int c : history) counts[c]++;
        return (int)(std::max_element(counts, counts + N_CLASSES) - counts);
    }
    bool is_ready() const { return (int)history.size() >= VOTE_WINDOW; }
};

// ── Alerta de dashboard ──────────────────────────────────────────
static int last_alerted_class = -1;

bool is_anomaly(int idx) {
    for (int i = 0; i < N_NORMAL; i++)
        if (strcmp(LABELS[idx], NORMAL_CLASSES[i]) == 0) return false;
    return true;
}

DWORD WINAPI alert_thread(LPVOID param) {
    int idx = *(int*)param;
    delete (int*)param;
    if (idx == last_alerted_class) return 0;
    last_alerted_class = idx;

    char msg[512];
    snprintf(msg, sizeof(msg),
        "ALERTA DEL VEHICULO\n"
        "===================\n\n"
        "%s\n\n"
        "Falla detectada: %s\n\n"
        "Lleve su vehiculo a revision.",
        ALERT_MESSAGES[idx], LABELS[idx]);

    MessageBoxA(NULL, msg, "DIAGNOSTICO DEL VEHICULO",
                MB_OK | MB_ICONWARNING | MB_SYSTEMMODAL);
    return 0;
}

void trigger_alert(int idx) {
    int* param = new int(idx);
    HANDLE t = CreateThread(NULL, 0, alert_thread, param, 0, NULL);
    if (t) CloseHandle(t);
}

// ── Serial ───────────────────────────────────────────────────────
HANDLE serial_open(const char* port, int baud) {
    char fp[32]; snprintf(fp, sizeof(fp), "\\\\.\\%s", port);
    HANDLE h = CreateFileA(fp, GENERIC_READ|GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "[ERROR] No se puede abrir %s\n", port); return h; }
    DCB dcb = {0}; dcb.DCBlength = sizeof(dcb);
    GetCommState(h, &dcb); dcb.BaudRate = baud; dcb.ByteSize = 8; dcb.StopBits = ONESTOPBIT; dcb.Parity = NOPARITY;
    SetCommState(h, &dcb);
    COMMTIMEOUTS t = {50, 10, 500, 0, 0}; SetCommTimeouts(h, &t);
    return h;
}

std::string serial_read_line(HANDLE h) {
    std::string line; char c; DWORD r;
    while (ReadFile(h, &c, 1, &r, NULL) && r > 0) { if (c=='\n') break; if (c!='\r') line+=c; }
    return line;
}

std::vector<double> parse_window(const std::string& line) {
    std::vector<double> s;
    size_t colon = line.find(':');
    if (colon == std::string::npos) return s;
    std::string data = line.substr(colon + 1);
    char* buf = &data[0], *tok = strtok(buf, ",");
    while (tok && (int)s.size() < N_SAMPLES) { s.push_back(atof(tok)); tok = strtok(nullptr, ","); }
    return s;
}

// ── ONNX ─────────────────────────────────────────────────────────
struct OnnxModel {
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::string input_name, output_name;

    OnnxModel(const char* path)
        : env(ORT_LOGGING_LEVEL_WARNING, "car"),
          session(env, std::wstring(path, path+strlen(path)).c_str(), Ort::SessionOptions{})
    {
        input_name  = session.GetInputNameAllocated(0, allocator).get();
        output_name = session.GetOutputNameAllocated(0, allocator).get();
    }

    int predict(const std::vector<float>& mfcc, float* probs_out) {
    std::array<int64_t, 2> shape = {1, N_MFCC};
    Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(
        mi, const_cast<float*>(mfcc.data()), mfcc.size(), shape.data(), 2);

    const char* in_n = input_name.c_str(), *out_n = output_name.c_str();
    auto out = session.Run(Ort::RunOptions{nullptr}, &in_n, &in, 1, &out_n, 1);
    float* scores = out[0].GetTensorMutableData<float>();

    // Temperature scaling — T=0.05 calibrado en Python
    const float T = 0.05f;

    float max_score = *std::max_element(scores, scores + N_CLASSES);
    float sum = 0.0f;
    for (int i = 0; i < N_CLASSES; i++) {
        probs_out[i] = expf((scores[i] - max_score) / T);
        sum += probs_out[i];
    }
    for (int i = 0; i < N_CLASSES; i++) probs_out[i] /= sum;

    int best = 0;
    for (int i = 1; i < N_CLASSES; i++)
        if (probs_out[i] > probs_out[best]) best = i;

    return best;
}
};

// ── Main ─────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Uso: %s <COM> <modelo.onnx>\n", argv[0]);
        return 1;
    }

    printf("\n==============================================\n");
    printf("  Car Diagnostic - Inferencia en tiempo real\n");
    printf("  Puerto : %s  |  Voto: %d ventanas\n", argv[1], VOTE_WINDOW);
    printf("==============================================\n\n");

    init_mel_filterbank();
    printf("[OK] Filtros Mel\n");

    OnnxModel model(argv[2]);
    printf("[OK] Modelo cargado\n");

    HANDLE serial = serial_open(argv[1], 115200);
    if (serial == INVALID_HANDLE_VALUE) return 1;
    printf("[OK] Conectado a %s\n\n", argv[1]);

    printf("%-10s %-30s %s\n", "Ventana", "Diagnostico", "Estado");
    printf("%s\n", std::string(55, '-').c_str());

    MajorityVoter voter;
    uint32_t count        = 0;
    int      last_reported = -1;

    while (true) {
        std::string line = serial_read_line(serial);
        if (line.empty() || line[0] != 'W') continue;

        auto samples = parse_window(line);
        if ((int)samples.size() != N_SAMPLES) continue;

        auto mag  = compute_fft_magnitudes(samples);
        auto mfcc = compute_mfcc(mag);

        // Obtener predicción y probabilidades
        float probs[N_CLASSES];
        int   pred = voter.vote(model.predict(mfcc, probs));

        if (voter.is_ready() && pred != last_reported) {
            bool anomaly = is_anomaly(pred);

            // Construir JSON con probabilidades incluidas
            char json[1024];
            int  pos = 0;
            pos += snprintf(json + pos, sizeof(json) - pos,
                "{\"is_fault\":%s,\"label\":\"%s\",\"probabilities\":{",
                anomaly ? "true" : "false", LABELS[pred]);

            for (int i = 0; i < N_CLASSES; i++) {
                pos += snprintf(json + pos, sizeof(json) - pos,
                    "\"%s\":%.4f%s",
                    LABELS[i], probs[i],
                    i < N_CLASSES - 1 ? "," : "");
            }

            pos += snprintf(json + pos, sizeof(json) - pos, "}}");

            // Enviar al puente WebSocket
            printf("%s\n", json);
            fflush(stdout);

            // Log legible en consola
            printf("%-10u %-30s %s\n", count, LABELS[pred],
                   anomaly ? "*** ANOMALIA ***" : "Normal");

            if (anomaly) trigger_alert(pred);
            last_reported = pred;
        }

        count++;
    }

    CloseHandle(serial);
    return 0;
}
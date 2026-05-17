// Libraries 
# include "arduinoFFT.h"

// Define sample dimensions 
# define SAMPLES 512
# define SAMPLING_FREQ 16000

// Arrays for FFT
double vReal[SAMPLES];
double vImag[SAMPLES];

// FFT object
arduinoFFT FFT = arduinoFFT(vReal, vImage, SAMPLES, SAMPLING_FREQ);

const int MIC_PIN = A0;

void setup(){
  Serial.begin(115200);
  while(!Serial);
}

void loop(){
  // Sliding windows 
  // Get samples
  unsigned long microseconds = micros();
  for (int i = 0; i < SAMPLES; i++) {
    vReal[i] = analogRead(MICRO_PIN);
    vImag[i] = 0;

    // Wait time to keep 16kHz
    while (micros() - microsecond < (1000000 / SAMPLING_FREQ)){

    }

    microseconds += (100000 / SAMPLING_FREQ);
  }

  // Apply Hann windows
  FFT.Windowing(FFT_WIN_TYP_HANN, FFT_FORWARD);
  
  // Compute FFT
  FF.Compute(FFT_FORWARD);

  // Calculate magnitude (Complex -> Reals)
  FFT.ComplexToMagnitude(); 

  // Send data to CPU
  for (int i = 2; i < (SAMPLES/2); i++){
    Serial.print(vReal[i]);
    Serial.print(",");
  }

  Serial.println();

  delay(10);
}

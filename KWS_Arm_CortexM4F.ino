// ===============================
// Macros to enable DSP / CMSIS-NN
// ===============================
#define ARM_MATH_DSP           // Enable DSP-optimized math (SIMD/FPU on Cortex-M4)
#define ARM_MATH_LOOPUNROLL    // Enable loop unrolling in CMSIS-DSP
#define TF_LITE_USE_CMSIS_NN   // Force TensorFlow Lite Micro to use CMSIS-NN kernels for int8

// ===============================
// Include necessary libraries
// ===============================
#include <Arduino.h>           // Core Arduino functions
#include <PDM.h>               // PDM audio
#include <arm_math.h>          // CMSIS-DSP math functions (FFT, filters, etc.)
#include <TensorFlowLite.h>    // Core TensorFlow Lite Micro
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/micro/micro_profiler.h>

// Model include
#include "kws_1class_depthwise_int8.h"
#include "mel_filter_coefs.h"

// =======================
// Parameters
// =======================
#define SAMPLE_RATE 16000
#define FRAME_SIZE 512
#define HOP_SIZE (FRAME_SIZE / 2)
#define PDM_BUFFER_SIZE (FRAME_SIZE / 2)
#define SPECTRUM_BINS (FRAME_SIZE / 2 + 1)
#define NUM_MEL 40
#define NUM_FRAMES 61
#define THRESHOLD 0.8f

// =======================
// Audio buffers
// =======================
int16_t pdmBuffer[PDM_BUFFER_SIZE];
int16_t audioBuffer[PDM_BUFFER_SIZE];

volatile bool dataReady = false;
volatile int frameCounter = 0;
volatile bool firstFrame = true;

float audioFrame[FRAME_SIZE];
float windowed[FRAME_SIZE];
float hannWindow[FRAME_SIZE];
float fftOut[FRAME_SIZE];
float mag[SPECTRUM_BINS];
int8_t quantized_mel_row[NUM_MEL];
int8_t featureBuffer[NUM_FRAMES * NUM_MEL];

// CMSIS-DSP matrix/vector instances
arm_matrix_instance_f32 melMat, magVec, melOut;
float melRowTmp[NUM_MEL];

// FFT instance
arm_rfft_fast_instance_f32 fft_inst;

// =======================
// TFLite variables
// =======================
constexpr int kTensorArenaSize = 96 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
static tflite::MicroProfiler profiler;

float input_scale;
int32_t input_zero_point;
float output_scale;
int32_t output_zero_point;

// =======================
// PDM callback
// =======================
void onPDMData() {
  int bytes = PDM.available();
  if (bytes <= 0) return;

  int samples = PDM.read(pdmBuffer, bytes) / 2;
  for (int i = 0; i < samples; i++)
    audioBuffer[i] = pdmBuffer[i];

  dataReady = true;
}

// =======================
// Compute Hann window
// =======================
void computeHannWindow() {
  for (int n = 0; n < FRAME_SIZE; n++)
    hannWindow[n] = 0.5f - 0.5f * cosf(2.0f * PI * n / (FRAME_SIZE - 1));
}

// =======================
// Compute log-Mel for one frame
// =======================
void computeLogMel() {
  arm_mult_f32(audioFrame, hannWindow, windowed, FRAME_SIZE);
  arm_rfft_fast_f32(&fft_inst, windowed, fftOut, 0);

  for (int i = 0; i < SPECTRUM_BINS; i++) {
    float real = fftOut[2*i];
    float imag = fftOut[2*i+1];
    mag[i] = real*real + imag*imag;
  }

  magVec.pData = mag;
  melOut.pData = melRowTmp;

  arm_mat_mult_f32(&melMat, &magVec, &melOut);

  for (int m = 0; m < NUM_MEL; m++) {
    float log_mel = logf(melRowTmp[m] + 1e-6f);
    int32_t q = (int32_t)roundf(log_mel / input_scale + input_zero_point);
    quantized_mel_row[m] = (int8_t)constrain(q, -128, 127);
  }

  memmove(featureBuffer, featureBuffer + NUM_MEL, (NUM_FRAMES - 1) * NUM_MEL * sizeof(int8_t));
  memcpy(featureBuffer + (NUM_FRAMES - 1) * NUM_MEL, quantized_mel_row, NUM_MEL * sizeof(int8_t));
}

// =======================
// Process audio buffer
// =======================
void processAudio() {
  if (firstFrame) {
    memset(audioFrame, 0, sizeof(audioFrame));
    firstFrame = false;
    for (int i = 0; i < HOP_SIZE; i++)
      audioFrame[i] = audioBuffer[i] / 32768.0f;
  } else {
    memmove(audioFrame, audioFrame + HOP_SIZE, HOP_SIZE * sizeof(float));
    for (int i = 0; i < HOP_SIZE; i++)
      audioFrame[i + HOP_SIZE] = audioBuffer[i] / 32768.0f;

    computeLogMel();
    frameCounter++;
  }
}

// =======================
// Arduino setup
// =======================
void setup() {
  Serial.begin(115200);
  while (!Serial);

  arm_rfft_fast_init_f32(&fft_inst, FRAME_SIZE);
  arm_mat_init_f32(&melMat, NUM_MEL, SPECTRUM_BINS, (float*)melFilterbank);
  arm_mat_init_f32(&magVec, SPECTRUM_BINS, 1, mag);
  arm_mat_init_f32(&melOut, NUM_MEL, 1, melRowTmp);

  computeHannWindow();

  PDM.onReceive(onPDMData);
  if (!PDM.begin(1, SAMPLE_RATE)) while (1);
  PDM.setGain(60);

  // Load TFLite model
  model = tflite::GetModel(kws_1class_depthwise_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) while (1);

  // =======================
  // Ops resolver (int8 CMSIS-NN will be used automatically)
  // =======================
  static tflite::MicroMutableOpResolver<4> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddSoftmax();
  resolver.AddMean(); // optional; reference kernel

  // Interpreter
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, nullptr, &profiler);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  input_scale = input->params.scale;
  input_zero_point = input->params.zero_point;
  output_scale = output->params.scale;
  output_zero_point = output->params.zero_point;

  Serial.println("Setup complete.");
}

// =======================
// Arduino loop
// =======================
void loop() {
  if (!dataReady) return;
  dataReady = false;
  processAudio();

  if (frameCounter >= NUM_FRAMES) {
    frameCounter = 0;
    firstFrame = true;

    memcpy(input->data.int8, featureBuffer, NUM_FRAMES * NUM_MEL * sizeof(int8_t));

    uint32_t t0 = millis();
    interpreter->Invoke();
    uint32_t t1 = millis();

    profiler.Log();
    Serial.print("Inference ms: ");
    Serial.println(t1 - t0);

    int8_t raw = output->data.int8[0];
    float prob = output_scale * (raw - output_zero_point);

    Serial.print("Raw output: ");
    Serial.print(raw);
    Serial.print(" Probability: ");
    Serial.print(prob);

    if(prob > THRESHOLD)
      Serial.print(" keyword DETECTED!");  
    Serial.println(" ");
  }
}


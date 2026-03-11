#include <Arduino.h>
#include <PDM.h>
#include "arm_math.h"

// TensorFlow Lite Micro
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// =======================
// Model include
// =======================
#include "kws_1class_depthwise_int8.h"
#include "mel_filter_coefs.h"

// =======================
// Audio & FFT/Mel parameters
// =======================
#define SAMPLE_RATE     16000
#define FRAME_SIZE      512
#define HOP_SIZE        256
#define PDM_BUFFER_BYTES 512 //only need to be 512 for arduino
#define PDM_BUFFER_SIZE 256   // PDM_BUFFER_BYTES / 2
#define CAPTURE_SIZE    (HOP_SIZE*(NUM_FRAMES-1) + FRAME_SIZE)//4096  // buffer for DMA burst
#define SPECTRUM_BINS   257
#define NUM_MEL         40
#define NUM_FRAMES      61

// =======================
// Audio buffers
// =======================
int16_t pdmBuffer[PDM_BUFFER_SIZE];
int16_t audioCapture[CAPTURE_SIZE];
volatile int captureIndex = 0;
volatile bool captureReady = false;

float audioFrame[FRAME_SIZE];
float windowed[FRAME_SIZE];
float hannWindow[FRAME_SIZE];
float fftOut[FRAME_SIZE];
float mag[SPECTRUM_BINS];
int8_t quantized_mel_row[NUM_MEL];
int8_t featureBuffer[NUM_FRAMES * NUM_MEL];


// CMSIS-DSP matrix/vector instances
arm_matrix_instance_f32 melMat;
arm_matrix_instance_f32 magVec;
arm_matrix_instance_f32 melOut;
// Temporary buffer for float energies
float melRowTmp[NUM_MEL];

// =======================
// TFLite variables
// =======================
constexpr int kTensorArenaSize = 96*1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

float input_scale;
int32_t input_zero_point;
float output_scale;
int32_t output_zero_point;

// FFT instance
arm_rfft_fast_instance_f32 fft_inst;

// =======================
// PDM callback: just collect samples
// =======================
void onPDMData() {
    int bytesAvailable = PDM.available();
    if (bytesAvailable <= 0) return;

    int samplesRead = PDM.read(pdmBuffer, bytesAvailable) / 2;
    for (int i = 0; i < samplesRead && captureIndex < CAPTURE_SIZE; i++) {
        audioCapture[captureIndex++] = pdmBuffer[i];
    }

    if (captureIndex >= CAPTURE_SIZE) {
        captureReady = true;
        captureIndex = 0;
    }
}

// =======================
// Precompute Hann window
// =======================
void computeHannWindow() {// just call this in setup without arm code for simplicity since one and done
    for(int n = 0; n < FRAME_SIZE; n++){
        hannWindow[n] = 0.5f - 0.5f * cosf(2.0f * PI * n / (FRAME_SIZE - 1));
    }
}

// =======================
// Compute log-Mel for a single frame
// =======================
void computeLogMel() {
    // Hanning Window of input audio
    arm_mult_f32(audioFrame, hannWindow, windowed, FRAME_SIZE);
    // FFT
    arm_rfft_fast_f32(&fft_inst, windowed, fftOut, 0);
    // Mag squared
    arm_cmplx_mag_squared_f32(fftOut, mag, SPECTRUM_BINS);

    //  Update CMSIS matrix pointers for this frame
    magVec.pData = mag;       // Always point to current FFT magnitude
    melOut.pData = melRowTmp; // Ensure output points to working buffer

    // Inputs:
    // mag[SPECTRUM_BINS]           - magnitude spectrum (float32)
    // melFilterbank[NUM_MEL][SPECTRUM_BINS] - mel filterbank coefficients (float32)
    // quantized_mel_row[NUM_MEL]   - output int8 row
    // input_scale, input_zero_point - quantization parameters

    // Perform matrix-vector multiply: energies for all Mel bins
    arm_mat_mult_f32(&melMat, &magVec, &melOut);

    // Log + quantization (per Mel bin)
    for(int m = 0; m < NUM_MEL; m++) {
        float log_mel = logf(melRowTmp[m] + 1e-6f);
        int32_t rounded = (int32_t)roundf((log_mel / input_scale) + (float)input_zero_point);
        quantized_mel_row[m] = (int8_t)constrain(rounded, -128, 127);
    }
    
    // Shift feature buffer left
    memmove(featureBuffer,
            featureBuffer + NUM_MEL,
            (NUM_FRAMES - 1) * NUM_MEL * sizeof(int8_t));

    // Copy new frame to last slot
    memcpy(featureBuffer + (NUM_FRAMES - 1) * NUM_MEL,
           quantized_mel_row,
           NUM_MEL * sizeof(int8_t));
}

// =======================
// Process captured audio buffer
// =======================
void processCapturedAudio() {
    int numFrames = (CAPTURE_SIZE - FRAME_SIZE) / HOP_SIZE + 1;

    for(int f = 0; f < numFrames; f++) {
        // Copy 512-sample frame
        for(int i = 0; i < FRAME_SIZE; i++) {
            audioFrame[i] = audioCapture[f * HOP_SIZE + i] / 32768.0f;
        }

        // Compute log-Mel
        computeLogMel();
    }
}

// =======================
// Arduino setup
// =======================
void setup() {
    Serial.begin(115200);
    while(!Serial);

    arm_rfft_fast_init_f32(&fft_inst, FRAME_SIZE);
    // Initialize matrices
    arm_mat_init_f32(&melMat, NUM_MEL, SPECTRUM_BINS, (float*)melFilterbank);  // NUM_MEL x SPECTRUM_BINS
    arm_mat_init_f32(&magVec, SPECTRUM_BINS, 1, mag);                           // SPECTRUM_BINS x 1
    arm_mat_init_f32(&melOut, NUM_MEL, 1, melRowTmp);                           // NUM_MEL x 1

    computeHannWindow();

    PDM.onReceive(onPDMData);
    if(!PDM.begin(1, SAMPLE_RATE)) {
        Serial.println("PDM failed");
        while(1);
    }
    PDM.setGain(60);

    model = tflite::GetModel(kws_1class_depthwise_int8_tflite);
    if(model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch");
        while(1);
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena,
        kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    if(interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors failed");
        while(1);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    input_scale = input->params.scale;
    input_zero_point = input->params.zero_point;

    output_scale = output->params.scale;
    output_zero_point = output->params.zero_point;

    Serial.println("Setup complete");
    Serial.println("KWS for keyword YES from Google's Speech Command database."); 
}

// =======================
// Arduino loop
// =======================
void loop() {
    if(captureReady) {
        captureReady = false;

        // Process all frames in captured buffer
        processCapturedAudio();

        memcpy(input->data.int8, featureBuffer, NUM_FRAMES * NUM_MEL * sizeof(int8_t));

        // Run inference
        if(interpreter->Invoke() != kTfLiteOk) {
            Serial.println("Invoke failed");
        } else {
            int8_t raw_output = output->data.int8[0];
            float probability = output_scale * (raw_output - output_zero_point);
            Serial.print("Raw output: "); 
            Serial.print((int)raw_output);
            Serial.print(" --- Probability: ");
            Serial.println(probability);  
        }
    }
}
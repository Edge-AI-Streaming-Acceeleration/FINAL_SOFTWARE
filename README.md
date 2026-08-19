# Edge AI Streaming Accelerator for ECG Arrhythmia Detection

## Overview

This project implements an **Edge AI Streaming Accelerator** for real-time ECG arrhythmia detection using a lightweight **1D Convolutional Neural Network (CNN)**.

The objective is to move ECG inference from cloud-based processing toward **local edge hardware**, reducing latency, communication overhead, and dependency on continuous cloud connectivity.

The software pipeline is developed in **PyTorch** and converted into an **INT8 quantized representation** suitable for implementation using synthesizable Verilog HDL.

---

## Project Architecture

```text
                 ECG Sensor
                     |
                     v
              ECG Input Samples
                     |
                     v
              +--------------+
              |   Conv1D #1  |
              |   1 -> 8     |
              +--------------+
                     |
                     v
                   ReLU
                     |
                     v
                Max Pool
                     |
                     v
              +--------------+
              |   Conv1D #2  |
              |   8 -> 16    |
              +--------------+
                     |
                     v
                   ReLU
                     |
                     v
                Max Pool
                     |
                     v
               Flatten (720)
                     |
                     v
              +--------------+
              |     FC1      |
              |   720 -> 32  |
              +--------------+
                     |
                     v
                   ReLU
                     |
                     v
              +--------------+
              |     FC2      |
              |    32 -> 1   |
              +--------------+
                     |
                     v
             Arrhythmia / Normal

Dataset

The model uses the MIT-BIH Arrhythmia Database.

ECG signals are segmented around annotated heartbeats and converted into fixed-length input vectors.

Input
Dataset: MIT-BIH Arrhythmia Database
Input type: ECG signal
Input length: 180 samples
Input channels: 1
Classification: Binary
Class 0: Normal
Class 1: Arrhythmia
CNN Architecture

The hardware-oriented CNN uses a deliberately lightweight architecture.

Layer	Configuration
Input	1 × 180
Conv1	1 → 8, Kernel = 5
ReLU	Integer ReLU
MaxPool1	Kernel = 2, Stride = 2
Conv2	8 → 16, Kernel = 5
ReLU	Integer ReLU
MaxPool2	Kernel = 2, Stride = 2
Flatten	720 values
FC1	720 → 32
ReLU	Integer ReLU
FC2	32 → 1
Output	Binary classification

The network is intentionally small to make it suitable for resource-constrained edge hardware.

INT8 Quantization

The trained FP32 model is converted to an INT8 quantized model.

Quantization is used to reduce:

Memory requirements
Multiplication complexity
Hardware resource usage
Power consumption
Data movement

The hardware pipeline operates primarily using integer arithmetic.

Quantization Components

The project uses:

INT8 activations
INT8 weights
INT32 accumulators
Per-channel weight quantization
Quantized biases
Layer-specific output scales
Layer-specific zero points
Verified Quantization Pipeline

The quantized software pipeline has been verified layer-by-layer against PyTorch.

Conv1
Input
  ↓
INT8 Conv1
  ↓
INT8 ReLU
  ↓
INT8 MaxPool
Conv2
INT8 Pool1
  ↓
INT8 Conv2
  ↓
INT8 ReLU
  ↓
INT8 MaxPool
Fully Connected Layers
720 INT8 values
       ↓
    FC1
       ↓
32 INT8 values
       ↓
     ReLU
       ↓
     FC2
       ↓
Classification
Golden Vector Verification

A major objective of this project is to create bit-accurate golden reference data for hardware verification.

The software generates expected outputs for each stage of the CNN.

The following intermediate results are generated:

input_ecg.hex


conv1_weights.hex
conv1_bias.hex
conv1_expected.hex


relu1_expected.hex
pool1_expected.hex


conv2_weights.hex
conv2_bias.hex
conv2_expected.hex


relu2_expected.hex
pool2_expected.hex


fc1_input_expected.hex
fc1_weights.hex
fc1_bias.hex
fc1_expected.hex
fc1_relu_expected.hex


fc2_weights.hex
fc2_bias.hex
fc2_expected.hex

These files can be used as reference vectors for Verilog simulation.

Bit-Accurate Verification

The quantized implementation was verified against PyTorch at individual layers.

Example verification:

CONV2 FULL EXACT COMPARISON


Total elements : 1440
Exact matches  : 1440
Mismatches     : 0
Match %        : 100.0
Max abs diff   : 0
Mean abs diff  : 0.0

FC1 verification:

FC1 PYTORCH vs GOLDEN


Max difference : 0
Mean difference: 0.0
Exact match    : 100.0 %

FC2 verification:

FC2 PYTORCH vs GOLDEN


PyTorch : [23]
Golden  : [23]


Difference: [0]
Exact match: 100.0 %

This establishes a software golden reference before RTL implementation.

Hardware Parameters

The hardware_params/ directory contains hexadecimal golden vectors and quantization information.

hardware_params/
├── conv1_weights.hex
├── conv1_bias.hex
├── conv1_expected.hex
├── conv1_weight_scales.txt
├── conv2_weights.hex
├── conv2_bias.hex
├── conv2_expected.hex
├── conv2_weight_scales.txt
├── fc1_weights.hex
├── fc1_bias.hex
├── fc1_expected.hex
├── fc1_input_expected.hex
├── fc1_relu_expected.hex
├── fc1_weight_scales.txt
├── fc2_weights.hex
├── fc2_bias.hex
├── fc2_expected.hex
├── fc2_weight_scales.txt
├── input_ecg.hex
├── pool1_expected.hex
├── pool2_expected.hex
├── relu1_expected.hex
├── relu2_expected.hex
└── quant_params.txt
Verilog Parameters

The verilog_int8_params/ directory contains files specifically formatted for hardware implementation.

verilog_int8_params/
├── conv1_weights.mem
├── conv1_bias_int32.mem
├── conv1_weight_scales.txt
├── conv1_weight_zero_points.txt
│
├── conv2_weights.mem
├── conv2_bias_int32.mem
├── conv2_weight_scales.txt
├── conv2_weight_zero_points.txt
│
├── fc1_weights.mem
├── fc1_bias_int32.mem
├── fc1_weight_scales.txt
├── fc1_weight_zero_points.txt
│
├── fc2_weights.mem
├── fc2_bias_int32.mem
├── fc2_weight_scales.txt
├── fc2_weight_zero_points.txt
│
├── input_quantization.txt
├── quantization_params.txt
└── hardware_quantization.txt

The .mem files can be loaded into Verilog memories using constructs such as:

$readmemh("conv1_weights.mem", conv1_weights);
Hardware-Oriented Design

The CNN is designed to map onto a streaming hardware architecture.

Potential RTL blocks include:

Input Buffer
     |
     v
Line Buffer
     |
     v
MAC / Convolution Engine
     |
     v
ReLU
     |
     v
Max Pooling
     |
     v
Convolution Engine
     |
     v
Fully Connected Engine
     |
     v
Classifier

The design avoids dependence on floating-point arithmetic during inference.

Why INT8?

Floating-point CNN inference is expensive for small edge devices.

INT8 inference provides a more hardware-friendly representation:

FP32
 |
 | Quantization
 v
INT8
 |
 +---- INT8 × INT8
 |
 v
INT32 Accumulator
 |
 | Requantization
 v
INT8

This allows the accelerator to use integer MAC units instead of floating-point processing units.

Software Environment

Recommended environment:

Python
PyTorch
NumPy
WFDB
Jupyter Notebook

Example installation:

pip install torch numpy wfdb jupyter
Repository Structure
FINAL_SOFTWARE/
│
├── README.md
│
├── newsplitting.ipynb
│
├── hardware_params/
│   ├── *.hex
│   ├── *.txt
│   └── golden vectors
│
└── verilog_int8_params/
    ├── *.mem
    ├── *.txt
    └── quantization parameters

Large training artifacts such as:

*.pth
*.npy
*.npz

are excluded from version control.

Verification Flow

The intended development flow is:

MIT-BIH ECG Dataset
        |
        v
FP32 CNN Training
        |
        v
FP32 Model Validation
        |
        v
INT8 Quantization
        |
        v
PyTorch INT8 Verification
        |
        v
Golden Vector Generation
        |
        v
Verilog RTL
        |
        v
RTL Simulation
        |
        v
Compare RTL Output
        |
        v
Bit-Accurate Hardware Verification
Project Goal

The final objective is to implement the trained and quantized CNN as a standalone streaming hardware accelerator capable of performing ECG arrhythmia inference locally.

The software repository provides the trained-model reference, quantization parameters, weights, biases, input vectors, and expected intermediate outputs required for RTL verification.

Future Work
Implement complete Verilog CNN accelerator
Integrate streaming input buffer
Implement hardware MAC array
Implement INT32 accumulation and requantization
Implement pipelined convolution
Implement hardware max pooling
Implement FC layers
Develop RTL testbench
Perform RTL vs Python bit-accurate verification
Synthesize RTL
Evaluate area, timing, and power
FPGA prototype
Optimize throughput and latency
Project Title

Edge AI Streaming Accelerator: A Standalone ML Logic CNN for Edge Networks

Application

Real-Time ECG Arrhythmia Detection on Edge Hardware

Technology
Machine Learning
+
INT8 Quantization
+
Digital VLSI
+
Verilog HDL
+
Edge AI
Status
Completed
 MIT-BIH ECG preprocessing
 CNN model development
 FP32 model training
 INT8 quantization
 Conv1 verification
 Conv2 verification
 FC1 verification
 FC2 verification
 Golden vector generation
 Hardware weight generation
 INT8 .mem file generation
In Progress
 Complete RTL accelerator
 RTL testbench
 Bit-accurate RTL verification
 Synthesis
 Area/timing/power evaluation
 FPGA implementation

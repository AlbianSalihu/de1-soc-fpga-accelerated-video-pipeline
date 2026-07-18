# DE1-SoC FPGA-Accelerated Video Pipeline

## Overview

This repository is a hardware-oriented edge-inference project targeting the
**Intel Cyclone V SE SoC** on the Terasic DE1-SoC board.

The goal is to run **AlexNet64Gray**, a compact convolutional neural network, using a
fully integer-only FPGA inference datapath backed by a verification workflow that
compares the RTL against a trusted Python reference model.

The arithmetic pipeline uses:

```text
uint8 activations
    ↓
int8 weights
    ↓
signed int32 accumulation
    ↓
signed int32 bias
    ↓
ReLU
    ↓
fixed-point requantization
    ↓
uint8 activations
```

No floating-point arithmetic is required in the inference datapath.

The project combines:

- PyTorch training and post-training quantization;
- FPGA-oriented parameter and test-vector export;
- bit-exact Python integer inference;
- synthesizable VHDL;
- GHDL-based directed verification;
- real-model-vector comparison between RTL and Python.

The quantization approach is inspired by:

> Jacob et al., *Quantization and Training of Neural Networks for Efficient
> Integer-Arithmetic-Only Inference*  
> [https://arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877)

---

## Current project status

### Completed

#### Machine-learning and export pipeline

The repository includes a working ML pipeline for:

- float-model training;
- post-training activation calibration;
- per-output-channel weight quantization;
- signed int8 weight generation;
- signed int32 bias generation;
- fixed-point requantization multiplier and shift generation;
- integer-only inference verification;
- FPGA parameter export;
- per-layer binary test-vector export.

#### Generic convolution RTL

The current completed hardware milestone is:

```text
hardware/rtl/layers/conv_layer.vhd
```

The convolution layer supports:

- unsigned 8-bit activation streaming;
- signed 8-bit weight streaming;
- signed 32-bit accumulation;
- signed 32-bit per-channel bias;
- ReLU;
- unsigned 32-bit per-channel requantization multipliers;
- unsigned 8-bit per-channel right shifts;
- round-half-up requantization;
- unsigned 8-bit saturation;
- configurable tensor dimensions;
- configurable square kernel size;
- configurable symmetric zero padding;
- configurable horizontal and vertical stride;
- multiple input channels;
- multiple output-channel groups;
- parallel output-channel lanes;
- activation-source stalls;
- weight-source stalls;
- output backpressure;
- stable raw and quantized outputs while stalled;
- trailing input-row draining;
- frame-completion signaling;
- automatic restart for the next frame.

#### Convolution verification

The convolution implementation has been verified using:

- directed unit-style VHDL testbenches;
- padding and stride traversal tests;
- multiple-input-channel tests;
- multiple-output-group tests;
- activation and weight source-gap tests;
- output-backpressure tests;
- bias, ReLU, rounding, and saturation tests;
- raw signed-int32 accumulator comparison;
- final uint8 output comparison;
- frame-completion and restart checks;
- real exported vectors for every convolution layer in AlexNet64Gray.

Current real-vector result:

```text
Convolution real-vector results
  Ran:    5 layers
  Passed: 5 layers
  Failed: 0 layers
```

The convolution datapath is therefore functionally complete and bit-exact in
simulation for the current model.

---

### Remaining work

The complete FPGA inference pipeline is not yet finished.

Remaining major blocks include:

- `maxpool_layer`;
- `fc_layer`;
- packed-output stream adapters;
- weight-memory provider;
- parameter-loading controller;
- external SDRAM interface;
- top-level pipeline integration;
- argmax and result-output logic;
- complete-network RTL simulation;
- Quartus synthesis;
- DSP and embedded-memory inference inspection;
- timing closure;
- on-board validation.

The current RTL has not yet been characterized with Quartus, so published resource
counts and throughput figures remain planning estimates rather than implementation
results.

---

## Repository layout

```text
.
├── ml/
│   ├── training
│   ├── calibration
│   ├── quantization
│   ├── parameter export
│   ├── test-vector export
│   └── integer inference verification
│
├── hardware/
│   ├── RTL implementation
│   ├── architecture documentation
│   └── FPGA resource and throughput planning
│
├── verification/
│   ├── VHDL testbenches
│   ├── GHDL simulation scripts
│   ├── real-model-vector tests
│   └── generated raw accumulator references
│
├── software/
│   └── planned HPS-side drivers and runtime
│
└── docs/
    ├── diagrams
    ├── design notes
    └── project documentation
```

Detailed documentation:

- [Machine-learning pipeline](ml/README.md)
- [Hardware architecture and implementation](hardware/README.md)

---

# Machine-learning pipeline

The ML pipeline is located in:

```text
ml/
```

It performs:

- float model training;
- post-training activation calibration;
- per-channel weight quantization;
- integer-only parameter generation;
- quantized inference verification;
- FPGA binary export;
- intermediate activation export;
- per-layer convolution-vector export.

---

## Model

The implemented network is:

```text
AlexNet64Gray
```

Input:

```text
1 × 64 × 64 grayscale image
```

Output:

```text
10 classes
```

The model contains:

```text
Conv1
ReLU
MaxPool1

Conv2
ReLU
MaxPool2

Conv3
ReLU

Conv4
ReLU

Conv5
ReLU
MaxPool3

FC6
ReLU

FC7
ReLU

FC8
```

The model is defined in:

```text
ml/src/models/alexnet64gray.py
```

The architecture was selected to:

- be large enough to exercise meaningful FPGA compute and memory constraints;
- remain manageable for simulation and implementation;
- provide several convolution dimensions and channel counts;
- avoid BatchNorm, simplifying integer inference;
- expose realistic on-chip versus external-memory tradeoffs.

---

# Machine-learning flow

## Option A — Run the complete ML pipeline

```bash
python -m ml.scripts.run_pipeline \
    --config ml/config/default.yaml \
    --device cuda
```

Edit:

```text
ml/config/default.yaml
```

to change training and export parameters.

Individual phases may be skipped:

```bash
python -m ml.scripts.run_pipeline \
    --run-id 0 \
    --skip-train \
    --skip-calibrate
```

---

## Option B — Run each phase manually

### 1. Download the dataset

```bash
python -m ml.scripts.download_mnist
```

---

### 2. Train the float model

```bash
python -m ml.src.train.train \
    --epochs 10 \
    --batch-size 128 \
    --device cuda
```

Typical outputs:

```text
ml/checkpoints/runN/best.pth
ml/checkpoints/runN/last.pth
ml/runs/runN/run_meta.json
ml/runs/runN/final_report.json
```

---

### 3. Calibrate activation scales

```bash
python -m ml.src.export.find_scales \
    --hook relu \
    --percentile 0.999
```

The command auto-detects the most recent run unless a run is selected explicitly.

Output:

```text
ml/outputs/runN/act_scales_sy.json
```

---

### 4. Quantize weights and compute FPGA parameters

```bash
python -m ml.src.export.quantize_weights \
    --s0 0.02
```

Outputs include:

```text
ml/outputs/runN/fpgaqparms.npz
ml/outputs/runN/fpgaqparms.json
```

The generated parameters include:

```text
signed int8 weights
signed int32 biases
unsigned fixed-point requantization multipliers
unsigned requantization shifts
activation-scale metadata
tensor dimensions
```

---

### 5. Verify integer inference accuracy

```bash
python -m ml.src.export.test_quantized_model \
    --s0 0.02
```

This executes the quantized model using integer arithmetic and compares its accuracy
with the float model.

The same arithmetic definitions are used by the RTL verification flow.

---

### 6. Export FPGA binaries

```bash
python -m ml.src.export.export_weights
```

The exported data is consumed by the hardware testbenches and future runtime
infrastructure.

Per-convolution-layer files include:

```text
<prefix>_in.bin
<prefix>_weights.bin
<prefix>_biases.bin
<prefix>_requant_m.bin
<prefix>_requant_r.bin
<prefix>_out.bin
```

These files provide:

```text
input activations
signed convolution weights
signed biases
requantization multipliers
requantization shifts
expected final uint8 outputs
```

---

# Hardware implementation

The FPGA RTL is located in:

```text
hardware/
```

The current implemented layer is:

```text
hardware/rtl/layers/conv_layer.vhd
```

The complete pipeline remains under development.

---

## Current `conv_layer` architecture

The convolution layer is one synthesizable VHDL architecture containing concurrent
worker processes for:

```text
initial activation-line filling
prime-row filling
stream-row filling
physical line-buffer writing
logical row rotation
active weight loading
MAC traversal
raw result holding
bias and requantization parameter loading
quantized result holding
output backpressure
vertical stride control
trailing-row draining
frame completion and restart
```

The implementation does not currently instantiate separate `line_buffer`,
`mac_array`, or `requant_unit` entities.

Those functions are implemented inside the verified `conv_layer` architecture.

---

## Convolution generics

```vhdl
G_C_IN    : positive;
G_C_OUT   : positive;
G_W_IN    : positive;
G_H_IN    : positive;
G_C_PAR   : positive;
G_KERNEL  : positive;
G_PADDING : natural;
G_STRIDE  : positive
```

`G_C_PAR` is the number of output channels calculated in parallel.

The number of output-channel groups is:

```text
G_C_OUT / G_C_PAR
```

The current implementation requires:

```text
G_C_OUT mod G_C_PAR = 0
```

---

## Activation input

Activations enter one unsigned byte at a time:

```vhdl
i_valid : in  std_logic;
i_ready : out std_logic;
i_data  : in  std_logic_vector(7 downto 0);
```

Input ordering is:

```text
input row
    → input column
        → input channel
```

A transfer occurs when:

```text
i_valid = 1
and
i_ready = 1
```

Padding values are generated internally and are not included in the activation
stream.

---

## Weight input

Weights enter one signed byte at a time:

```vhdl
i_weight_valid : in  std_logic;
o_weight_ready : out std_logic;
i_weight_data  : in  std_logic_vector(7 downto 0);
```

The active weight buffer contains:

```text
G_C_PAR × G_KERNEL × G_KERNEL
```

weights.

This is one weight slice for:

```text
one output-channel group
one input channel
```

The full layer weight tensor is not stored inside `conv_layer`.

A future weight provider will supply slices from:

```text
on-chip RAM
external SDRAM
HPS DDR3
or another streaming source
```

---

## Bias and requantization parameters

Per-output-channel parameters are loaded through:

```vhdl
cfg_we    : in std_logic;
cfg_sel   : in std_logic_vector(1 downto 0);
cfg_addr  : in std_logic_vector(19 downto 0);
cfg_wdata : in std_logic_vector(31 downto 0);
```

Selections:

```text
cfg_sel = "01" → signed int32 bias
cfg_sel = "10" → unsigned uint32 requantization multiplier
cfg_sel = "11" → unsigned uint8 requantization shift
cfg_sel = "00" → reserved
```

Parameters remain stored across automatic frame restarts.

---

## Quantized output

Final uint8 activations are emitted in packed output groups:

```vhdl
o_valid : out std_logic;
o_data  : out std_logic_vector(G_C_PAR * 8 - 1 downto 0);
```

Each transfer contains:

```text
G_C_PAR output channels
```

Output ordering is:

```text
output row
    → output-channel group
        → output column
            → lane
```

The absolute output channel represented by lane `n` is:

```text
output_group × G_C_PAR + n
```

---

## Raw accumulator output

The current layer retains a raw signed-int32 output for debugging and verification:

```vhdl
i_acc_ready : in  std_logic;
o_acc_valid : out std_logic;
o_acc_data  : out std_logic_vector(G_C_PAR * 32 - 1 downto 0);
```

This output contains the convolution result before:

```text
bias
ReLU
requantization
saturation
```

The quantized and raw outputs represent the same tensor position.

The current implementation uses:

```text
i_acc_ready
```

as the ready input for both interfaces.

When the consumer applies backpressure, both packed outputs remain stable.

---

## Convolution arithmetic

For each output channel:

```text
raw_acc =
    Σ(uint8 activation × int8 weight)

biased_acc =
    raw_acc + signed int32 bias

relu_acc =
    max(biased_acc, 0)

product =
    uint64(relu_acc) × uint32(requant_multiplier)

if requant_shift > 0:
    product += 2^(requant_shift - 1)

shifted =
    product >> requant_shift

output =
    min(shifted, 255)
```

The rounding rule is round-half-up for nonnegative values.

There is currently no output zero-point addition.

---

## Line-buffer organization

The layer uses:

```text
G_KERNEL active physical rows
one spare physical row
```

for a total of:

```text
G_KERNEL + 1 physical row buffers
```

Logical row roles are rotated without copying activation contents.

For a 3×3 convolution:

```text
before:
    active rows = A, B, C
    spare row   = D

after:
    active rows = B, C, D
    spare row   = A
```

Virtual top and bottom padding rows are represented through row-valid flags.

Left and right padding values are generated when the calculated input-column index
falls outside the real image.

---

## Frame completion

The layer provides:

```vhdl
o_done : out std_logic;
```

`o_done` pulses for one clock after:

```text
the final output has been accepted
all required trailing input rows have been drained
the final logical row traversal has completed
```

The layer then returns to its initial state and automatically prepares for the next
frame.

A global reset is not required between frames.

The physical line-buffer memories do not need to be cleared because real rows are
overwritten before reuse and virtual rows are controlled through validity flags.

---

# RTL verification

The RTL verification flow is located in:

```text
verification/
```

It uses GHDL to compile and simulate the convolution layer.

---

## Directed convolution tests

Run:

```bash
bash verification/sim/run_conv.sh
```

The directed tests cover:

```text
initial line filling
prime-row filling
weight loading
concurrent activation and weight preparation
calculation
horizontal window sliding
logical line rotation
multiple output groups
output backpressure
frame completion
automatic restart
```

---

## Traversal matrix

Run:

```bash
bash verification/sim/run_conv_traversal.sh
```

The traversal tests cover combinations of:

```text
padding
stride 2
stride 3
multiple input channels
multiple output groups
activation source gaps
weight source gaps
backpressure
trailing-row draining
```

---

## Real-model-vector tests

Run all five convolution layers:

```bash
bash verification/sim/run_layers.sh
```

Run one selected layer:

```bash
bash verification/sim/run_layers.sh features_0
```

Regenerate raw accumulator references:

```bash
FORCE_GOLDEN=1 \
bash verification/sim/run_layers.sh
```

Enable waveform generation:

```bash
WAVE=1 \
bash verification/sim/run_layers.sh features_0
```

The real-vector testbench compares:

```text
o_acc_data
    against generated signed-int32 raw convolution references

o_data
    against exported uint8 model outputs
```

Generated raw references are stored under:

```text
verification/results/default/raw_acc/
```

A passing test therefore verifies both:

```text
window traversal and raw accumulation
```

and:

```text
bias, ReLU, requantization, rounding, and saturation
```

---

# FPGA implementation status

The current convolution RTL has passed functional simulation.

The following implementation work has not yet been performed:

```text
Quartus synthesis
DSP inference inspection
M10K and MLAB inference inspection
ALM and register measurement
place-and-route
TimeQuest timing analysis
Fmax measurement
power analysis
on-board validation
```

Any existing resource and throughput calculations in the repository should be treated
as pre-implementation estimates until Quartus reports are available.

---

# Planned complete pipeline

The intended top-level data path is:

```text
input image
    ↓
Conv1
    ↓
MaxPool1
    ↓
Conv2
    ↓
MaxPool2
    ↓
Conv3
    ↓
Conv4
    ↓
Conv5
    ↓
MaxPool3
    ↓
FC6
    ↓
FC7
    ↓
FC8
    ↓
argmax
    ↓
predicted class
```

Additional infrastructure will include:

```text
packed-lane serializers
inter-stage FIFOs
weight provider
parameter loader
external-memory controller
global pipeline controller
result interface
```

The packed convolution output must be converted into the next layer's expected
row-column-channel input order.

---

# What this project enables

This project is not primarily about training MNIST.

It is about developing and verifying reusable techniques for:

- FPGA-accelerated neural-network inference;
- deterministic low-latency edge processing;
- integer-only compute pipelines;
- hardware-aware quantization;
- streaming convolution architectures;
- custom memory and weight-delivery systems;
- RTL-to-software numerical equivalence;
- automated real-model-vector verification.

The same architecture and workflow can be adapted to:

- CNN-based video filters;
- object-detection accelerators;
- embedded computer vision;
- industrial inspection;
- real-time sensor processing;
- low-power edge-AI systems;
- custom FPGA inference engines.

---

# Project milestone summary

Completed:

```text
float training
activation calibration
weight quantization
integer parameter generation
integer inference verification
FPGA binary export
real convolution test-vector export
generic convolution RTL
padding and stride
multiple input channels
multiple output groups
bias and ReLU
fixed-point requantization
rounding and saturation
backpressure
frame completion
automatic restart
directed RTL regression
all five real convolution layers passing bit-exact comparison
```

Still in progress:

```text
pooling RTL
fully connected RTL
weight-memory subsystem
stream adapters
top-level integration
complete-network simulation
Quartus implementation
on-board deployment
```
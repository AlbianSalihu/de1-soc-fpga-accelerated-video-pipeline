# Hardware — FPGA Inference Pipeline

This directory contains the RTL implementation of a fully integer-only CNN inference pipeline
targeting the **Intel Cyclone V SE SoC** on the Terasic DE1-SoC board.

The pipeline runs **AlexNet64Gray** — a compact convolutional network trained and quantized
in the `ml/` directory — as a spatial streaming datapath entirely on the FPGA fabric.
All arithmetic is integer-only: int8 weights, uint8 activations, int32 accumulators,
with fixed-point requantisation between layers. No floating-point hardware is required.

---

## Pre-Implementation Analysis

Before writing a single line of VHDL, a full pre-implementation analysis was carried out
to verify that the design fits the hardware and to make every major architectural decision
on firm numerical ground rather than intuition.

The analysis is structured as six sequential steps, each building on the previous one:
resource inventory → compute requirements → throughput model → memory sizing →
module architecture → implementation plan.

**Key outcomes at a glance:**

| Question | Answer |
|----------|--------|
| How many parallel int8 MACs are available? | **220** (87 DSP blocks × 3, at 80% utilisation) |
| How much on-chip BRAM is available? | **496 KB** (397 M10K blocks) |
| Do all network weights fit in BRAM? | **No** — 19.2 MB total; only Conv1, Conv2, FC8 fit (120.5 KB) |
| What is the v1 target throughput? | **~7 fps** (FC weight streaming from SDRAM is the bottleneck) |
| What could v2 achieve with HPS offload of FC? | **~30 fps** (conv pipeline is DSP-bound at that point) |
| Does the datapath infrastructure fit in BRAM? | **Yes** — 163 of 397 M10K blocks used (41%) |
| Do sliding windows and accumulators fit in registers? | **Yes** — 56% of flip-flops used |

---

## Table of Contents

- [Step 1 — Hardware Baseline](#step-1--hardware-baseline)
- [Step 2 — Network MAC Requirements & Weight Storage](#step-2--network-mac-requirements--weight-storage)
- [Step 3 — Throughput Analysis & DSP Allocation](#step-3--throughput-analysis--dsp-allocation)
- [Step 4 — Memory Architecture & BRAM Verification](#step-4--memory-architecture--bram-verification)
- [Step 5 — RTL Block Diagram & Interface Definitions](#step-5--rtl-block-diagram--interface-definitions)
- [Step 6 — Implementation Plan & Build Order](#step-6--implementation-plan--build-order)

---

## Step 1 — Hardware Baseline

The first step was to establish the exact resource budget of the target device from the
official Intel datasheet, rather than relying on marketing figures. This matters because
synthesis tools work with ALMs and DSP blocks, not "logic elements" — a discrepancy
that commonly trips up early resource estimates.

**Target device:** Intel Cyclone V SE SoC — `5CSEMA5F31C6N`

`5CSEMA5F31C6N` is the full ordering code. The base silicon designation used in Intel's
product table is **5CSEA5** — confirmed by Altera's own product page:
*"Cyclone® V SE FPGA 5CSEA5 (F31) — 5CSEMA5F31C6N"*
([source](https://www.altera.com/products/fpga/cyclone/v/se/5csea5-f31/5CSEMA5F31C6N)).

| Field | Value | Meaning |
|-------|-------|---------|
| `5C` | Cyclone V | Device family |
| `SE` | SE variant | SoC with integrated dual-core ARM Cortex-A9 HPS |
| `A5` | Density code | 85K LEs (marketing), 32,070 ALMs (actual) |
| `F31` | Package | 896-pin FBGA, 31×31 mm |
| `C6` | Speed/temp | Commercial grade, speed grade 6 |
| `N` | Packaging | Lead-free / RoHS — no effect on fabric resources |

The `M` between `SE` and `A5` is an Altera catalog marker; it does not indicate a
different silicon variant. All fabric resources are identical to the base 5CSEA5.

---

### 1.1 Resource Table

All figures are from the Intel Cyclone V Product Table, row **"Cyclone V SE SoCs → 5CSEA5"**
([Cyclone V FPGA and SoC FPGA Product Table — Intel, v2025.09.24](https://cdrdv2-public.intel.com/714207/cyclone-v-product-table.pdf)):

| Resource | Count | Notes |
|----------|-------|-------|
| LEs (Logic Elements) | 85K | Marketing figure |
| ALMs (Adaptive Logic Modules) | **32,070** | True synthesizable unit |
| Registers | 128,300 | Flip-flops inside ALMs |
| M10K BRAM blocks | **397** | 10 Kb each |
| M10K total memory | **3,970 Kb** | ≈ **496 KB** |
| MLAB distributed memory | 480 Kb | LUT-based SRAM inside ALMs |
| Variable-precision DSP blocks | **87** | One physical block each |
| 18×18 multipliers | 174 | 87 × 2 — cross-checks DSP mode table |
| PLLs (FPGA fabric) | 6 | |
| HPS DDR3 SDRAM | 1 GB | Via HPS hard memory controller |
| Dedicated SDRAM (IS42S16320F) | 64 MB | 16-bit bus, ~150 MB/s, FPGA-accessible |

> The "85K LEs" figure from Terasic is a marketing aggregate.
> The actual synthesizable unit is the **ALM** — 32,070 of them.
> All resource estimates in this analysis use the ALM count.

---

### 1.2 DSP Block Modes

Each of the 87 DSP blocks is **variable-precision**: a single block can be reconfigured
at synthesis time to run in one of three modes. This is what makes the Cyclone V
particularly efficient for integer neural network inference.

| Mode | Multiplier size | Multiplications per block | Applicable to |
|------|----------------|--------------------------|---------------|
| 9×9 SIMD | 9-bit × 9-bit | **3** | int8 weights × uint8 activations |
| 18×18 | 18-bit × 18-bit | **2** | Standard integer multiply |
| 27×27 | 27-bit × 27-bit | **1** | High-precision (not needed here) |

The 9×9 SIMD mode is directly applicable to this design: int8 weights (−127 to +127)
and uint8 post-ReLU activations (0 to 255) both fit within 9 bits, so one DSP block
can perform **three independent MACs simultaneously**.

The 18×18 entry in the product table (174 = 87 × 2) independently cross-checks this figure.

---

### 1.3 Maximum Parallel MACs

| DSP mode | MACs per block | Total blocks | Max parallel int8 MACs |
|----------|---------------|--------------|------------------------|
| 9×9 SIMD | 3 | 87 | **261** |
| 18×18 | 2 | 87 | **174** |

A "MAC" here is one multiply-accumulate: `acc += weight(int8) × activation(uint8)`.
The int32 accumulator lives in fabric registers — not inside the DSP block itself.

Synthesis tools and routing overhead consume some DSP capacity. A practical working
budget of **80–85%** is used for all subsequent calculations:

| DSP mode | Practical budget |
|----------|-----------------|
| 9×9 SIMD | **~220 MACs** |
| 18×18 | ~140–150 MACs |

All throughput analysis uses the 9×9 SIMD budget of **220 parallel MACs**.

---

### 1.4 BRAM Budget

Each M10K block stores 10 Kbits = **1,280 bytes**. With 397 blocks, the total on-chip
memory is **496 KB**. This is the hard upper bound for on-chip weight storage —
layers whose weights exceed this must stream from the 64 MB external SDRAM.
The per-layer weight breakdown is computed in Step 2.

---

## Step 2 — Network MAC Requirements & Weight Storage

With the hardware budget established, the next step was to profile the network itself:
how much compute does each layer require, and how much memory do its weights occupy?
These figures drive every subsequent decision — DSP allocation, BRAM vs SDRAM placement,
and which layers dominate throughput.

The network is **AlexNet64Gray** (`ml/src/models/alexnet64gray.py`).
Input: 1×64×64 grayscale. Output: 10 classes.

---

### 2.1 Per-Layer Dimensions

| Layer | Input shape | Output shape | Kernel | Stride | Pad |
|-------|------------|-------------|--------|--------|-----|
| Conv1 | 1×64×64 | 64×64×64 | 5×5 | 1 | 2 |
| MaxPool1 | 64×64×64 | 64×32×32 | 2×2 | 2 | — |
| Conv2 | 64×32×32 | 192×32×32 | 3×3 | 1 | 1 |
| MaxPool2 | 192×32×32 | 192×16×16 | 2×2 | 2 | — |
| Conv3 | 192×16×16 | 384×16×16 | 3×3 | 1 | 1 |
| Conv4 | 384×16×16 | 256×16×16 | 3×3 | 1 | 1 |
| Conv5 | 256×16×16 | 256×16×16 | 3×3 | 1 | 1 |
| MaxPool3 | 256×16×16 | 256×8×8 | 2×2 | 2 | — |
| FC6 | 16,384 | 1,024 | — | — | — |
| FC7 | 1,024 | 1,024 | — | — | — |
| FC8 | 1,024 | 10 | — | — | — |

---

### 2.2 MACs per Image

```
Conv layer:  MACs = C_out × H_out × W_out × (C_in × K_h × K_w)
FC layer:    MACs = C_out × C_in
MaxPool:     no MACs — compare-and-select only
```

| Layer | Calculation | MACs |
|-------|-------------|------|
| Conv1 | 64 × 64×64 × (1×5×5) | **6.6M** |
| Conv2 | 192 × 32×32 × (64×3×3) | **113M** |
| Conv3 | 384 × 16×16 × (192×3×3) | **170M** |
| Conv4 | 256 × 16×16 × (384×3×3) | **226M** |
| Conv5 | 256 × 16×16 × (256×3×3) | **151M** |
| FC6 | 1,024 × 16,384 | **16.8M** |
| FC7 | 1,024 × 1,024 | **1.0M** |
| FC8 | 10 × 1,024 | **<0.1M** |
| **Total** | | **≈ 684M MACs / image** |

Conv3 + Conv4 + Conv5 alone account for **547M MACs — 80% of all compute**.
These three layers determine the throughput ceiling of the pipeline.

---

### 2.3 Weight Storage

```
weights (bytes) = C_out × C_in × K_h × K_w    [int8, 1 byte each]
biases  (bytes) = C_out × 4                    [int32, 4 bytes each]
```

| Layer | Weights | Biases | Total | Fits in 496 KB BRAM? |
|-------|---------|--------|-------|----------------------|
| Conv1 | 64×1×5×5 = 1,600 B | 256 B | **1.8 KB** | ✓ |
| Conv2 | 192×64×3×3 = 110,592 B | 768 B | **108.7 KB** | ✓ |
| Conv3 | 384×192×3×3 = 663,552 B | 1,536 B | **649.5 KB** | ✗ exceeds entire BRAM |
| Conv4 | 256×384×3×3 = 884,736 B | 1,024 B | **865.0 KB** | ✗ |
| Conv5 | 256×256×3×3 = 589,824 B | 1,024 B | **577.0 KB** | ✗ |
| FC6 | 1,024×16,384 = 16,777,216 B | 4,096 B | **16.0 MB** | ✗ 32× over budget |
| FC7 | 1,024×1,024 = 1,048,576 B | 4,096 B | **1.0 MB** | ✗ |
| FC8 | 10×1,024 = 10,240 B | 40 B | **10.0 KB** | ✓ |
| **Total** | | | **≈ 19.2 MB** | |

The network is **39× larger than available BRAM**. This immediately establishes that
most weights must live in external SDRAM and stream into the pipeline at runtime.

---

### 2.4 On-Chip vs SDRAM Placement

| Layer | Weight size | Placement | Reason |
|-------|------------|-----------|--------|
| Conv1 | 1.8 KB | **BRAM** | Tiny; accessed every input pixel |
| Conv2 | 108.7 KB | **BRAM** | Fits within budget |
| Conv3 | 649.5 KB | **SDRAM** | Exceeds the full BRAM budget alone |
| Conv4 | 865.0 KB | **SDRAM** | |
| Conv5 | 577.0 KB | **SDRAM** | |
| FC6 | 16.0 MB | **SDRAM** | |
| FC7 | 1.0 MB | **SDRAM** | |
| FC8 | 10.0 KB | **BRAM** | Negligible; fast output stage |

**BRAM allocation after weight placement:**

| Usage | Size |
|-------|------|
| Conv1 + Conv2 + FC8 weights & biases | 120.5 KB |
| Remaining for line buffers, FIFOs, buffers | **375.5 KB** |
| Total BRAM | 496 KB |

The 375.5 KB remainder is available for the streaming datapath infrastructure.
Step 4 verifies this is sufficient.

---

### 2.5 MACs per Output Pixel

This per-layer figure is used in Step 3 to derive DSP allocation.
It represents the work required per spatial position, per output channel.

```
MACs per output pixel = C_in × K_h × K_w
```

| Layer | MACs / output pixel |
|-------|---------------------|
| Conv1 | 1×5×5 = **25** |
| Conv2 | 64×3×3 = **576** |
| Conv3 | 192×3×3 = **1,728** |
| Conv4 | 384×3×3 = **3,456** |
| Conv5 | 256×3×3 = **2,304** |

Conv4 is the most compute-intensive layer per output pixel and sets the throughput floor.

---

## Step 3 — Throughput Analysis & DSP Allocation

With the compute profile of the network established, the next step was to determine
how fast the pipeline can run and how to allocate the 220 available DSPs across layers
to maximise throughput.

**v1 design choice:** all layers — including the FC layers — run entirely on the FPGA.
No HPS offload. This keeps the first implementation simple and self-contained.
Known performance limitations are documented in §3.5 with a clear path to improvement.

---

### 3.1 Throughput Model

The hardware uses a **spatial pipeline**: all layers are instantiated simultaneously
and the image streams through them in order, like an assembly line.
For the pipeline to be balanced, every stage must finish one image in the same number
of cycles. If one stage is slower than the rest, it becomes the bottleneck and the
entire pipeline runs at its rate.

Given a fixed DSP budget `D` and allocating DSPs proportionally to each layer's
MAC count, every layer achieves the same cycles-per-image and:

```
T_total = sum(all MACs) / D
P_i     = total_MACs_i × D / sum(all MACs)    (DSPs for layer i)
```

---

### 3.2 Per-Layer DSP Allocation

Using the 220-MAC practical budget (9×9 SIMD mode, 80% utilisation):

| Layer | Total MACs | Share | DSPs | Cycles / image |
|-------|-----------|-------|------|----------------|
| Conv1 | 6.6M | 1.0% | **2** | 3,277K |
| Conv2 | 113M | 16.5% | **36** | 3,146K |
| Conv3 | 170M | 24.8% | **55** | 3,089K |
| Conv4 | 226M | 33.0% | **73** | 3,101K |
| Conv5 | 151M | 22.0% | **48** | 3,146K |
| FC6 | 16.8M | 2.5% | **5** | 3,356K |
| FC7 + FC8 | ~1M | <0.2% | — | shared with FC6 |
| **Total** | **685M** | 100% | **219** | |

FC7 and FC8 round to zero DSPs at this scale. In practice they time-multiplex
with FC6's MAC array.

**Pipeline bottleneck: Conv1 at 3,277K cycles.**

```
T_total  = 3,277,000 cycles
At 100 MHz → 32.8 ms per image → ~30 fps  (if memory bandwidth were not a constraint)
```

---

### 3.3 SDRAM Bandwidth Check

The DSP analysis alone suggests ~30 fps. However, layers using SDRAM-stored weights
must read those weights from the external memory chip during each image pass.
This introduces a bandwidth constraint that must be checked separately.

| Data | Size | Time at 150 MB/s |
|------|------|-----------------|
| Conv3 weights | 649.5 KB | 4.3 ms |
| Conv4 weights | 865.0 KB | 5.8 ms |
| Conv5 weights | 577.0 KB | 3.8 ms |
| **Conv subtotal** | **2.09 MB** | **13.9 ms** |
| FC6 weights | 16.0 MB | 106.7 ms |
| FC7 weights | 1.0 MB | 6.7 ms |
| **FC subtotal** | **17.0 MB** | **113.3 ms** |

The conv weight streaming (13.9 ms) fits inside the 32.8 ms compute window —
the memory controller can prefetch weights while the DSPs are already computing. No problem there.

The FC layers are a different story. The FC6 weight matrix alone is 16 MB.
At 150 MB/s that takes 107 ms to read — while the actual MAC computation for FC
would take only ~30 ms with 5 DSPs. The FC stage is **~4× memory-bandwidth-bound**,
not compute-bound.

**Effective v1 throughput:**
```
T = 32.8 ms (conv, DSP-bound) + 113.3 ms (FC, bandwidth-bound)
  ≈ 146 ms per image  →  ~7 fps
```

---

### 3.4 Known Limitations & Future Optimisations

These are documented here as a clear upgrade path once v1 is working end-to-end.

| # | Optimisation | Expected gain | Complexity |
|---|-------------|---------------|-----------|
| 1 | **Offload FC to HPS ARM** — Cortex-A9 with NEON does ~4 MACs/cycle; FC6 completes in ~5 ms, overlapped with FPGA processing the next frame | Full conv pipeline at ~30 fps | Medium — HPS↔FPGA handshake |
| 2 | **Stream FC weights via DDR3 (HPS)** — 1 GB DDR3 has far higher bandwidth; keeps FC on FPGA but removes the bandwidth wall | ~4–8× FC bandwidth | Medium — Avalon/AXI bridge |
| 3 | **Replace FC with Global Average Pooling** — eliminates FC6/FC7 entirely; small linear classifier replaces them | Near-zero FC bandwidth cost | High — requires retraining |
| 4 | **Weight double-buffering** — DMA pre-fetches FC weights for frame N+1 while frame N is being processed | Partially hides weight-load latency | Low–Medium |

**v1 target: ~7 fps, end-to-end on FPGA hardware.**
**v2 target: ~30 fps via option 1 or 2 above.**

---

## Step 4 — Memory Architecture & BRAM Verification

Before committing to the architecture, every piece of on-chip memory used by the
streaming datapath was sized and totalled to confirm the design fits within the
available 397 M10K blocks. The analysis also accounts for fabric flip-flop usage,
since sliding window registers are a non-trivial consumer.

All activations are uint8 (1 byte). All weights are int8 (1 byte). Biases int32 (4 bytes).
Each M10K block = 10 Kbits = **1,280 bytes**.

**Five memory categories:**
1. Line buffers — store buffered rows so each conv layer can form its K×K sliding window
2. MaxPool row buffers — hold one input row for the 2×2 max comparison
3. On-chip weight banks — BRAM storage for Conv1, Conv2, FC8
4. FC6 input activation buffer — flattened Pool3 output, held while FC layers run
5. SDRAM weight FIFOs — small burst buffers per SDRAM-mapped layer to absorb SDRAM latency

---

### 4.1 Line Buffers

As pixels arrive row by row, a K×K conv layer must hold K−1 complete rows in BRAM
to form the sliding window at each position.

```
line_buffer_bytes = (K − 1) × W_in × C_in
```

| Layer | Kernel | Buffered rows | W_in | C_in | Bytes | M10K |
|-------|--------|--------------|------|------|-------|------|
| Conv1 | 5×5 | 4 | 64 | 1 | 256 B | 1 |
| Conv2 | 3×3 | 2 | 32 | 64 | 4,096 B | 4 |
| Conv3 | 3×3 | 2 | 16 | 192 | 6,144 B | 5 |
| Conv4 | 3×3 | 2 | 16 | 384 | 12,288 B | 10 |
| Conv5 | 3×3 | 2 | 16 | 256 | 8,192 B | 7 |
| | | | | **Subtotal** | **30.5 KB** | **27** |

---

### 4.2 MaxPool Row Buffers

A 2×2 max pool compares pixels from two consecutive input rows.
It buffers one complete input row while waiting for the row below it to arrive.

```
pool_buffer_bytes = W_in × C_in
```

| Pool | Input from | W_in | C_in | Bytes | M10K |
|------|-----------|------|------|-------|------|
| Pool1 | Conv1 output | 64 | 64 | 4,096 B | 4 |
| Pool2 | Conv2 output | 32 | 192 | 6,144 B | 5 |
| Pool3 | Conv5 output | 16 | 256 | 4,096 B | 4 |
| | | | **Subtotal** | **14.3 KB** | **13** |

---

### 4.3 On-Chip Weight Banks

```
weight_bank_bytes = (C_out × C_in × K_h × K_w) + (C_out × 4)
                     int8 weights                  int32 biases
```

| Layer | Weights | Biases | Total | M10K |
|-------|---------|--------|-------|------|
| Conv1 | 64×1×5×5 = 1,600 B | 256 B | 1,856 B | 2 |
| Conv2 | 192×64×3×3 = 110,592 B | 768 B | 111,360 B | 87 |
| FC8 | 10×1,024 = 10,240 B | 40 B | 10,280 B | 9 |
| | | **Subtotal** | **120.5 KB** | **98** |

Conv2 alone consumes 87 M10K blocks — the single largest consumer at 22% of the total
BRAM budget. This is unavoidable given the weight tensor size.

---

### 4.4 FC6 Input Activation Buffer

After Pool3, the 256×8×8 activation map (16,384 values) must be held in BRAM
while the FC stages process the full vector sequentially.

| Buffer | Size | M10K |
|--------|------|------|
| FC6 input (flattened Pool3 output) | 16,384 B (16 KB) | 13 |

---

### 4.5 SDRAM Weight Streaming FIFOs

Each SDRAM-mapped layer has a small on-chip FIFO that absorbs burst latency from the
external memory controller, decoupling it from the MAC pipeline's cycle-accurate timing.
Each FIFO holds one filter slice — the weights for a single output channel — ensuring
the MAC array always has the next set of weights ready without stalling.

| Layer | FIFO holds | Size | M10K |
|-------|-----------|------|------|
| Conv3 | one 192×3×3 filter | 1,728 B | 2 |
| Conv4 | one 384×3×3 filter | 3,456 B | 3 |
| Conv5 | one 256×3×3 filter | 2,304 B | 2 |
| FC6 | burst cap | 4,096 B | 4 |
| FC7 | one FC row | 1,024 B | 1 |
| | **Subtotal** | **12.6 KB** | **12** |

---

### 4.6 Sliding Window Registers

The active K×K pixel neighbourhood used each cycle to form the convolution operand
lives in fabric flip-flops, not BRAM. BRAM has a 1–2 cycle read latency and cannot
provide the simultaneous arbitrary access the MAC array needs every clock cycle.
Registers are zero-latency and fully parallel.

```
sliding_window_FFs = K_h × K_w × C_in × 8
```

| Layer | Window size | Flip-flops |
|-------|------------|-----------|
| Conv1 | 5×5×1 | 200 |
| Conv2 | 3×3×64 | 4,608 |
| Conv3 | 3×3×192 | 13,824 |
| Conv4 | 3×3×384 | **27,648** |
| Conv5 | 3×3×256 | 18,432 |
| **Subtotal** | | **64,712 FFs** |

Conv4's sliding window alone consumes 27,648 FFs. This is the tightest register
constraint in the design, though it remains within budget.

**Int32 accumulators** — one per parallel MAC unit, also in fabric registers:

| Layer | Parallel MACs | FFs |
|-------|--------------|-----|
| Conv1 | 2 | 64 |
| Conv2 | 36 | 1,152 |
| Conv3 | 55 | 1,760 |
| Conv4 | 73 | 2,336 |
| Conv5 | 48 | 1,536 |
| FC6 | 5 | 160 |
| **Subtotal** | | **7,008 FFs** |

---

### 4.7 BRAM Tally — Go / No-Go

| Category | M10K blocks | KB |
|----------|------------|-----|
| Line buffers | 27 | 33.8 |
| MaxPool row buffers | 13 | 16.3 |
| On-chip weight banks | 98 | 122.5 |
| FC6 activation buffer | 13 | 16.3 |
| SDRAM weight FIFOs | 12 | 15.0 |
| **Total used** | **163** | **203.8** |
| **Available** | **397** | **496.2** |
| **Remaining headroom** | **234** | **292.5** |

**BRAM: GO.** 41% utilisation. 234 M10K blocks remain for inter-stage FIFOs,
control registers, and any additions during implementation.

---

### 4.8 Flip-Flop Tally — Go / No-Go

| Category | FFs | % of 128,300 |
|----------|-----|-------------|
| Sliding window registers | 64,712 | 50.4% |
| Int32 accumulators | 7,008 | 5.5% |
| **Total committed** | **71,720** | **55.9%** |
| **Remaining for control/pipeline** | **56,580** | **44.1%** |

**Registers: GO.** 44% of flip-flops remain for control FSMs, pipeline stage registers,
requantisation logic, and SDRAM controller state.

---

## Step 5 — RTL Block Diagram & Interface Definitions

With the resource analysis complete, the next step was to define every RTL module,
its interface, and how they connect — before writing any VHDL. This blueprint ensures
that the implementation phase is free to focus on correctness rather than architecture.

A key design principle throughout is **parameterisation**: every layer module is fully
generic, driven by constants in the top-level instantiation. Changing a layer's channel
count, kernel size, or parallelism requires only editing the generic map — no internal
RTL changes.

---

### 5.1 Module Hierarchy

```
fpga_pipeline_top
│
├── conv_layer          [G_LAYER=1]   Conv1  1→64   5×5  BRAM weights  P=2
│   ├── line_buffer                   4 rows × 64B
│   ├── sliding_window                5×5×1  registers
│   ├── mac_array                     2 parallel MACs
│   ├── requant_unit                  (m,r) multiply+shift → uint8
│   └── weight_bram                   1,856 B
│
├── maxpool_layer       [G_LAYER=1]   Pool1  64ch  64→32px
│   └── row_buffer                    4,096 B
│
├── conv_layer          [G_LAYER=2]   Conv2  64→192  3×3  BRAM weights  P=36
│   ├── line_buffer                   2 rows × 4,096B
│   ├── sliding_window                3×3×64  registers
│   ├── mac_array                     36 parallel MACs
│   ├── requant_unit
│   └── weight_bram                   111,360 B
│
├── maxpool_layer       [G_LAYER=2]   Pool2  192ch  32→16px
│   └── row_buffer                    6,144 B
│
├── conv_layer          [G_LAYER=3]   Conv3  192→384  3×3  SDRAM weights  P=55
│   ├── line_buffer                   2 rows × 6,144B
│   ├── sliding_window                3×3×192  registers
│   ├── mac_array                     55 parallel MACs
│   ├── requant_unit
│   └── weight_fifo                   1,728 B  ← fed by sdram_ctrl
│
├── conv_layer          [G_LAYER=4]   Conv4  384→256  3×3  SDRAM weights  P=73
│   ├── line_buffer                   2 rows × 12,288B
│   ├── sliding_window                3×3×384  registers
│   ├── mac_array                     73 parallel MACs
│   ├── requant_unit
│   └── weight_fifo                   3,456 B
│
├── conv_layer          [G_LAYER=5]   Conv5  256→256  3×3  SDRAM weights  P=48
│   ├── line_buffer                   2 rows × 8,192B
│   ├── sliding_window                3×3×256  registers
│   ├── mac_array                     48 parallel MACs
│   ├── requant_unit
│   └── weight_fifo                   2,304 B
│
├── maxpool_layer       [G_LAYER=3]   Pool3  256ch  16→8px
│   └── row_buffer                    4,096 B
│
├── activation_buffer                 16,384 B  (flattened Pool3 → FC input)
│
├── fc_layer            [G_LAYER=6]   FC6  16384→1024  SDRAM weights  P=5
│   ├── mac_array                     5 parallel MACs
│   ├── requant_unit
│   └── weight_fifo                   4,096 B
│
├── fc_layer            [G_LAYER=7]   FC7  1024→1024  SDRAM weights  P=5
│   ├── mac_array                     5 parallel MACs
│   ├── requant_unit
│   └── weight_fifo                   1,024 B
│
├── fc_layer            [G_LAYER=8]   FC8  1024→10  BRAM weights  P=5
│   ├── mac_array                     5 parallel MACs
│   └── weight_bram                   10,280 B
│
├── argmax_unit                       finds winning class from 10 logits
├── sdram_ctrl                        Avalon-MM master — streams weights to layer FIFOs
└── pipeline_ctrl                     global flow control (sop/eop, stall, frame sync)
```

---

### 5.2 Inter-Stage Streaming Interface (Avalon-ST)

Every connection between pipeline stages uses the same streaming interface.
Data flows as one uint8 activation per cycle, one channel at a time, in row-major order.
The `channel` signal is carried alongside the data so downstream modules can route
activations to the correct accumulator without maintaining an external counter.

```
pixel_stream:
  valid   : std_logic                      -- data on bus is valid this cycle
  ready   : std_logic                      -- downstream ready to accept (backpressure)
  sop     : std_logic                      -- first activation of a new image
  eop     : std_logic                      -- last activation of an image
  channel : std_logic_vector(11 downto 0)  -- output channel index (0..C_out-1)
  data    : std_logic_vector( 7 downto 0)  -- uint8 activation value
```

**Flow rule:** a stage must not assert `valid` until its line buffer is fully populated
(all K×K×C_in values present). At startup and at image boundaries, the line buffer
asserts `window_valid` internally to gate the MAC array.

---

### 5.3 Layer Generics

**conv_layer**
```vhdl
entity conv_layer is
  generic (
    G_C_IN      : positive;           -- input channels
    G_C_OUT     : positive;           -- output channels
    G_H_IN      : positive;           -- input height (pixels)
    G_W_IN      : positive;           -- input width  (pixels)
    G_KERNEL    : positive := 3;      -- kernel size (3 or 5)
    G_PADDING   : natural  := 1;      -- zero-padding each side
    G_PAR_MACS  : positive;           -- parallel MAC units
    G_WEIGHT_SRC: string   := "BRAM"  -- "BRAM" or "SDRAM"
  );
```

**maxpool_layer**
```vhdl
entity maxpool_layer is
  generic (
    G_CHANNELS  : positive;
    G_W_IN      : positive;
    G_H_IN      : positive;
    G_POOL_SIZE : positive := 2;
    G_STRIDE    : positive := 2
  );
```

**fc_layer**
```vhdl
entity fc_layer is
  generic (
    G_C_IN      : positive;
    G_C_OUT     : positive;
    G_PAR_MACS  : positive;
    G_WEIGHT_SRC: string := "SDRAM"
  );
```

---

### 5.4 Sub-Module Interfaces

**line_buffer**
```
Inputs:  clk, rst_n, din (uint8), wr_en, col_idx
Outputs: window_out [K×K×C_in] of uint8,  window_valid
```

**mac_array**
```
Inputs:  clk, rst_n, window [K×K×C_in] of uint8,
         weights [P][K×K×C_in] of int8, bias [P] of int32, start, weight_sel
Outputs: acc_out [P] of int32,  acc_valid
```

**requant_unit** — integer-only: `q_out = clip(((acc × m) >> r), 0, 255)`
```
Inputs:  acc (int32), m (uint32), r (uint8)
Outputs: q_out (uint8)
```

**sdram_ctrl** — Avalon-MM master; feeds all layer weight FIFOs from external SDRAM
```
Inputs:  clk, rst_n, req_layer, req_addr, req_len
Outputs: avmm_addr/read/rddata/rdvalid/waitreq,  fifo_data, fifo_wr_en, fifo_full
```

---

### 5.5 Top-Level Port Map

```
Inputs:
  clk_100     : std_logic              -- 100 MHz system clock
  rst_n       : std_logic              -- active-low reset

  -- pixel stream (from HPS or camera interface)
  pix_valid, pix_sop, pix_eop : std_logic
  pix_data    : std_logic_vector(7 downto 0)   -- uint8 grayscale

  -- SDRAM chip (IS42S16320F, 16-bit bus)
  sdram_dq    : std_logic_vector(15 downto 0)
  sdram_addr  : std_logic_vector(12 downto 0)
  sdram_ba    : std_logic_vector(1 downto 0)
  sdram_cas_n, sdram_ras_n, sdram_we_n, sdram_cs_n, sdram_clk, sdram_cke : std_logic
  sdram_dqm   : std_logic_vector(1 downto 0)

Outputs:
  result_valid : std_logic
  result_class : std_logic_vector(3 downto 0)   -- class index 0..9
  result_conf  : std_logic_vector(7 downto 0)   -- raw logit of winning class
```

---

### 5.6 Planned RTL File Layout

```
hardware/rtl/
├── top/
│   └── fpga_pipeline_top.vhd
├── layers/
│   ├── conv_layer.vhd
│   ├── maxpool_layer.vhd
│   └── fc_layer.vhd
├── primitives/
│   ├── line_buffer.vhd
│   ├── mac_array.vhd
│   ├── requant_unit.vhd
│   ├── weight_bram.vhd
│   ├── weight_fifo.vhd
│   └── argmax_unit.vhd
├── memory/
│   └── sdram_ctrl.vhd
└── control/
    └── pipeline_ctrl.vhd
```

---

### 5.7 V1/V2 Modularity Strategy

The architecture is designed so that v2 improvements — HPS offload, dynamic weight loading,
master control — require **wiring changes at the top level only**. No layer module is
ever rewritten between versions.

This is achieved by separating two independent concerns inside each layer:

#### Data plane vs control plane

**Data plane** — always on FPGA, unchanged between versions:
- Activations stream between layers via Avalon-ST (valid/ready/data)
- Each layer has one streaming input and one streaming output
- The MAC array, line buffer, and requant unit are purely data-driven

**Control plane** — evolves between versions:
- V1: no master; weights initialised from `.mif` files at power-on; port B tied off
- V2: a master (HPS ARM or FPGA state machine) writes weights at runtime via port B

#### Dual-port BRAM weight storage (internal implementation detail)

Weights are **internal** to each layer — nothing crosses the layer boundary for weights
in V1. The layer is a black box with only data in and data out:

```
         ┌─────────────────────────────────────┐
         │              conv_layer              │
         │                                     │
data ────►  in                           out ──►  data
         │                                     │
         │   ┌─────────────┐                  │
         │   │ weight BRAM │                  │
         │   │  port A ────┼──► MAC array     │
         │   │  port B     │  (tied off, V1)  │
         │   └─────────────┘                  │
         │   line buffer, sliding window, ...  │
         └─────────────────────────────────────┘
```

Inside the BRAM, port A is read every cycle by the MAC array. Port B exists physically
but is tied off in V1 — no ports are exposed on the layer boundary.

In V2, port B gets wired up: it becomes actual input ports on the layer module,
connected to a master via Avalon-MM. That change is local to the layer's port list
and the top-level wiring — the internal MAC array and compute logic are untouched.

#### V2 address map

When port B is exposed in V2, each layer is assigned a fixed base address on the
Avalon-MM bus. The master routes weight writes by address — no layer ID signal needed.

| Layer | Base address | Weight size |
|-------|-------------|-------------|
| Conv1 | 0x0000_0000 | 2 KB |
| Conv2 | 0x0000_1000 | 112 KB |
| Conv3–5 | SDRAM-mapped | — |
| FC8 | 0x0002_0000 | 11 KB |

#### Interface width and parallelism

The Avalon-ST data bus between layers carries **one uint8 activation per cycle**.
Internal parallelism is controlled by the `G_PAR_MACS` generic — it sets how many
MAC units run simultaneously inside a layer, completely independent of the external
interface width.

- The inter-layer interface stays narrow and simple (8-bit)
- Each layer scales its internal compute independently via `G_PAR_MACS`
- Increasing parallelism in one layer never affects any other layer's interface

---

## Step 6 — Implementation Plan & Build Order

The implementation follows a strict **bottom-up, simulation-first** discipline:
every primitive is written and verified in isolation before being composed into a layer,
and every layer passes its testbench before being connected to the top-level.
Nothing advances to the next phase until the current phase has zero failures.

---

### 6.1 Test Vector Strategy

The Python quantized model (`ml/src/export/test_quantized_model.py`) already executes
full integer inference with access to every intermediate tensor. Before any VHDL is
written, it is extended to dump golden test vectors for each module:

```
ml/outputs/runN/test_vectors/
├── input_image.bin          -- 64×64 uint8 pixel values
├── conv1_weights.bin        -- int8 weights + int32 biases
├── conv1_in.bin             -- uint8 activations entering Conv1
├── conv1_out.bin            -- uint8 activations leaving Conv1 (post-requant)
├── conv1_acc_sample.bin     -- int32 accumulators for a known pixel position
├── ...                      -- same pattern for conv2..5, pool1..3, fc6..8
└── final_class.txt          -- expected argmax output
```

Every VHDL testbench reads these files as stimulus and compares output bit-for-bit
against the golden values. A mismatch is a **hard failure**, not a warning.
This directly ties hardware verification to the Python reference model and makes any
numerical discrepancy immediately traceable to a specific layer and operation.

---

### 6.2 Simulation Toolchain

| Tool | Role |
|------|------|
| **GHDL** | Open-source VHDL simulator — fast compile, scriptable, no licence required |
| **GTKWave** | Waveform viewer for GHDL output |
| **ModelSim** (Quartus Lite) | Waveform debugging for complex timing issues |
| **Quartus Prime Lite** | Synthesis, place-and-route, resource and timing reports |
| **Python** | Test vector generation and result comparison scripts |

Quartus is only opened for resource reports and final bitstream generation.
All functional verification happens in simulation first.

---

### 6.3 Phase 1 — Primitives

The six primitives have no inter-dependencies and can be developed in parallel.

| # | Module | Test criteria |
|---|--------|--------------|
| 1 | `requant_unit` | Known (acc, m, r) triples from Python → correct uint8 output. Saturation at 0 and 255. |
| 2 | `mac_array` | Conv1 weights + known 5×5×1 window → int32 accumulator matches Python exactly. All-zero input check. |
| 3 | `line_buffer` | Full 64-wide row streamed in → correct 5×5 window at the right cycle. First and last pixel boundary cases. |
| 4 | `weight_bram` | Write known weights, read back. Verify 1-cycle M10K read latency. |
| 5 | `weight_fifo` | Fill to capacity → `full` flag asserts. Drain → `empty` flag asserts. Simultaneous read/write. |
| 6 | `argmax_unit` | All 10 FC8 logits → correct class index. Tie-breaking behaviour. |

**Exit criterion:** all six testbenches pass with zero mismatches against Python vectors.

---

### 6.4 Phase 2 — Memory Controller

| # | Module | Test criteria |
|---|--------|--------------|
| 7 | `sdram_ctrl` | Behavioural SDRAM model (IS42S16320F). Burst read of Conv3 weight block fills `weight_fifo` correctly. SDRAM refresh cycles do not corrupt in-flight data. |

The SDRAM controller is the most complex individual module — it must handle burst
timing, CAS latency, refresh arbitration, and FIFO backpressure simultaneously.
Extra debug time should be budgeted here.

**Exit criterion:** 1,728 bytes of Conv3 weights read from simulated SDRAM,
deposited into `weight_fifo`, verified byte-for-byte against the Python export file.

---

### 6.5 Phase 3 — Layers

Layers are built in increasing complexity, using progressively wider generics.

| # | Module | Configuration | Key challenge |
|---|--------|--------------|--------------|
| 8 | `conv_layer` | Conv1 — BRAM, P=2, 5×5 | First integration: line_buffer + mac_array + requant. Validate pipeline timing end-to-end. |
| 9 | `maxpool_layer` | Pool1 — 64ch, 64→32px | Row-buffer timing: row N must be held until row N+1 arrives. |
| 10 | `conv_layer` | Conv2 — BRAM, P=36, 3×3 | 36 parallel accumulators must stay synchronised through the full image. |
| 11 | `maxpool_layer` | Pool2 — 192ch | Same as Pool1 at wider channel count. |
| 12 | `conv_layer` | Conv3 — SDRAM, P=55, 3×3 | First SDRAM-fed layer. sdram_ctrl + weight_fifo enter the critical path. |
| 13 | `conv_layer` | Conv4, Conv5 | Same pattern as Conv3, different generics. |
| 14 | `maxpool_layer` | Pool3 — 256ch | |
| 15 | `fc_layer` | FC6, FC7 — SDRAM | No sliding window. Verify sequential neuron accumulation against Python. |
| 16 | `fc_layer` | FC8 — BRAM | Final logits feed argmax_unit. |

**Exit criterion per layer:** complete output tensor matches Python golden file
for all spatial positions and channels of one full test image.

---

### 6.6 Phase 4 — Integration & On-Board Verification

| # | Task | Success criteria |
|---|------|-----------------|
| 17 | Assemble `fpga_pipeline_top` | Compiles cleanly, no unconnected ports, no bus-width mismatches |
| 18 | End-to-end simulation — 1 image | `result_class` matches Python argmax |
| 19 | End-to-end simulation — 10 images | All 10 match; no state leakage between frames (tests sop/eop flush logic) |
| 20 | Quartus synthesis | DSP, M10K, ALM counts within pre-calculated budgets |
| 21 | Quartus place-and-route | All timing paths close at 100 MHz (Fmax ≥ 100 MHz) |
| 22 | On-board test | Correct class for 10 known MNIST images, on hardware, verified via UART or LED display |

---

### 6.7 Dependency Graph

```
requant_unit  ──────────────────────────────────────────┐
mac_array     ──────────────────────────────────────────┤
line_buffer   ──────────────────────────────────────────┼──► conv_layer (BRAM)  ──┐
weight_bram   ──────────────────────────────────────────┘                         │
                                                                                  │
weight_fifo   ──────┐                                                             │
sdram_ctrl    ──────┴───────────────────────────────────── conv_layer (SDRAM) ──┤
                                                                                  │
maxpool_layer (row_buffer only) ─────────────────────────────────────────────────┤
fc_layer      ───────────────────────────────────────────────────────────────────┤
argmax_unit   ───────────────────────────────────────────────────────────────────┤
activation_buffer ───────────────────────────────────────────────────────────────┤
pipeline_ctrl ───────────────────────────────────────────────────────────────────┤
                                                                                  ▼
                                                                    fpga_pipeline_top
```

---

### 6.8 Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| SDRAM timing closure at 100 MHz | Medium | High | Pipeline register at SDRAM output; fall back to 80 MHz target if needed |
| Conv4 sliding window (27,648 FFs) causes place-and-route congestion | Low | Medium | Split into two sub-stages sharing the window; or store half the window in BRAM with a 2-cycle accumulate schedule |
| SDRAM refresh stall exceeds weight FIFO depth, causing MAC pipeline to starve | Medium | Medium | Size FIFOs to absorb worst-case refresh duration (≤ 7.8 µs = 780 cycles at 100 MHz) |
| requant intermediate (acc × m) overflows int64 before the right-shift | Low | High | Pre-verified in Python before export; overflow assertion added to testbench |
| Frame-to-frame state leak through unflushed line buffers | Medium | High | Explicit flush FSM triggered on eop; testbench always runs ≥ 2 consecutive images |

# Verification

This directory contains the full verification infrastructure for the FPGA inference pipeline.

The approach is **co-simulation**: Python generates known-good intermediate values from
the integer reference model, VHDL testbenches consume those values and simulate the hardware,
and Python comparison scripts validate the results. No floating-point is involved at any stage.

A mismatch anywhere in the chain is a hard failure — not a warning, not a tolerance check.
The hardware must produce bit-exact results against the Python reference model.

---

## Table of Contents

- [Workflow Overview](#workflow-overview)
- [Folder Structure](#folder-structure)
- [How to Run](#how-to-run)
- [Data Formats](#data-formats)
- [Verification Phases](#verification-phases)
- [Adding a New Testbench](#adding-a-new-testbench)

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python side                              │
│                                                                 │
│  1. Save a fixture image (one-time)                             │
│       verification/fixtures/default/                            │
│           image.png            ← 64×64 grayscale, viewable      │
│           image_raw_u8.bin     ← raw bytes fed to hardware      │
│           meta.json            ← true/predicted class, run_id  │
│                                                                 │
│  2. Dump binary test vectors (re-run when model changes)        │
│       verification/vectors/default/                             │
│           features_0_in.bin    ← Conv1 input  (int8)           │
│           features_0_out.bin   ← Conv1 output (uint8)          │
│           features_0_weights.bin, _biases.bin                   │
│           features_0_requant.json, _requant_m.bin, _requant_r.bin│
│           features_0_acc_sample.bin                             │
│           ... same pattern for every layer ...                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │  read vectors/
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        VHDL side (GHDL)                         │
│                                                                 │
│  tb/primitives/tb_requant_unit.vhd                              │
│  tb/layers/tb_conv_layer.vhd                                    │
│  tb/pipeline/tb_fpga_pipeline_top.vhd                           │
│       │                                                         │
│       │  assert on mismatch (hard stop)                         │
│       │  write simulation output                                │
│       ▼                                                         │
│  results/default/                                               │
│       requant_out.bin, conv_layer_out.bin, ...                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │  read results/
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Python side                              │
│                                                                 │
│  compare/compare_all.py                                         │
│       reads vectors/ (expected) + results/ (actual)            │
│       computes per-layer error stats                            │
│       prints pass / fail report                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Two-stage validation is intentional:**
- The VHDL testbench asserts immediately on any mismatch → fast stop during simulation
- The Python comparison script runs after simulation → richer report with error statistics,
  worst-case deviations, and per-channel breakdowns for debugging

---

## Folder Structure

```
verification/
│
├── README.md                       ← this file
│
├── generate/
│   └── dump_vectors.py             ← Python entry point; reads ml/ outputs, writes fixtures/ and vectors/
│
├── fixtures/                       ← static test images (PNG + bin tracked by git, .bin ignored)
│   └── default/
│       ├── image.png               ← 64×64 grayscale PNG (human-readable)
│       ├── image_raw_u8.bin        ← same image as raw uint8 bytes (gitignored)
│       └── meta.json               ← true class, predicted class, run_id
│
├── vectors/                        ← generated binary vectors (fully gitignored)
│   └── default/
│       ├── features_0_in.bin       ← Conv1 input (int8, H×W×C)
│       ├── features_0_out.bin      ← Conv1 output (uint8, H×W×C)
│       ├── features_0_weights.bin  ← int8 weights
│       ├── features_0_biases.bin   ← int32 biases
│       ├── features_0_requant.json ← m/r params (human-readable)
│       ├── features_0_requant_m.bin← uint32 multipliers (VHDL-readable)
│       ├── features_0_requant_r.bin← uint8 shift amounts (VHDL-readable)
│       ├── features_0_acc_sample.bin← int32 accumulator at centre pixel
│       └── ... same pattern for every layer ...
│
├── tb/                             ← VHDL testbenches
│   ├── primitives/                 ← one testbench per primitive module
│   │   ├── tb_requant_unit.vhd
│   │   ├── tb_mac_array.vhd
│   │   ├── tb_line_buffer.vhd
│   │   ├── tb_weight_bram.vhd
│   │   ├── tb_weight_fifo.vhd
│   │   └── tb_argmax_unit.vhd
│   ├── layers/                     ← one testbench per layer type
│   │   ├── tb_conv_layer.vhd
│   │   ├── tb_maxpool_layer.vhd
│   │   └── tb_fc_layer.vhd
│   └── pipeline/                   ← end-to-end testbench
│       └── tb_fpga_pipeline_top.vhd
│
├── results/                        ← simulation outputs (fully gitignored)
│   └── default/
│       └── *.bin
│
├── compare/                        ← Python: reads vectors/ and results/, reports pass/fail
│   ├── compare_layer.py
│   └── compare_all.py
│
└── sim/                            ← GHDL runner scripts
    ├── run_all.sh
    ├── run_primitives.sh
    ├── run_layers.sh
    └── run_pipeline.sh
```

---

## How to Run

### 1. Save the default fixture image (one-time)

```bash
python -m verification.generate.dump_vectors --save-fixture
```

Finds the first correctly-classified test image, saves it to `verification/fixtures/default/`.
Only needs to be re-run if you want a different image.

### 2. Dump binary test vectors

```bash
python -m verification.generate.dump_vectors --dump-vectors
```

Runs the full quantized forward pass on the fixture image and writes one set of binary
files per layer to `verification/vectors/default/`.
Re-run this any time the model or quantization parameters change.

### 3. Both steps at once

```bash
python -m verification.generate.dump_vectors --save-fixture --dump-vectors
```

### 4. Run primitive testbenches (Phase 1)

```bash
bash verification/sim/run_primitives.sh
```

Each testbench prints `PASS` or `FAIL` and exits with the appropriate code.

### 5. Run layer testbenches (Phase 3)

```bash
bash verification/sim/run_layers.sh
```

### 6. Run end-to-end pipeline testbench

```bash
bash verification/sim/run_pipeline.sh
```

### 7. Run Python comparison report

```bash
python -m verification.compare.compare_all
```

### 8. Run everything

```bash
bash verification/sim/run_all.sh
```

---

## Data Formats

### Binary files (`.bin`)

All binary files are raw packed arrays with no header.

| File suffix | Data type | Layout | Notes |
|---|---|---|---|
| `_in.bin` (Conv) | int8 / uint8 | H×W×C row-major | int8 for first layer only |
| `_out.bin` (Conv/Pool) | uint8 | H×W×C row-major | channel-last for VHDL streaming |
| `_in.bin` (FC) | uint8 | (C,) flat | |
| `_out.bin` (FC) | uint8 | (C,) flat | |
| `_weights.bin` | int8 | C_out×C_in×KH×KW | C-contiguous |
| `_biases.bin` | int32 LE | (C_out,) | |
| `_acc_sample.bin` | int32 LE | (C_out,) | Conv: centre pixel; FC: full vector |
| `_acc.bin` | int32 LE | (C_out,) | Last FC only — no requant applied |
| `_requant_m.bin` | uint32 LE | (C_out,) | Fixed-point multiplier per channel |
| `_requant_r.bin` | uint8 | (C_out,) | Right-shift amount per channel |

### JSON files (`.json`)

Small metadata files tracked by git.

| File | Contents |
|---|---|
| `fixtures/default/meta.json` | True class, predicted class, run_id, image shape |
| `vectors/default/*_requant.json` | m[] and r[] per channel (human-readable copy of the .bin files) |

---

## Verification Phases

| Phase | Scope | Entry point |
|---|---|---|
| 1 | Primitives (requant, MAC, line buffer, ...) | `sim/run_primitives.sh` |
| 2 | Memory controller (sdram_ctrl) | `sim/run_layers.sh` |
| 3 | Individual layers (conv, pool, FC) | `sim/run_layers.sh` |
| 4 | End-to-end pipeline | `sim/run_pipeline.sh` |

**Exit criterion for each phase:** zero assertion failures in GHDL,
zero mismatches in the Python comparison report.

---

## Adding a New Testbench

1. In `generate/dump_vectors.py`, confirm the vectors you need are already written
   (every compute layer writes `_in`, `_out`, `_weights`, `_biases`, `_acc_sample`, `_requant_*`)
2. Create `tb/<category>/tb_<module_name>.vhd`
   - Read stimulus from `vectors/default/<prefix>_*.bin`
   - Write output to `results/default/<module>_out.bin`
   - Assert bit-exact match against the corresponding `_out.bin` vector
3. Add a GHDL compile + run entry to the appropriate `sim/run_*.sh` script
4. Add a comparison call in `compare/compare_all.py`

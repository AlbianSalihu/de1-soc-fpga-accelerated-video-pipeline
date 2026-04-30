# conv_layer

Generic convolutional layer for the FPGA inference pipeline.

---

## Role

Performs one convolution + ReLU + requantisation stage of the CNN.
Accepts a streaming pixel input, produces a streaming activation output.
Fully self-contained — no external control signals, no external weight bus in V1.

---

## Interface

```
Generics:
  G_C_IN      : positive           -- input channels
  G_C_OUT     : positive           -- output channels
  G_H_IN      : positive           -- input height (pixels)
  G_W_IN      : positive           -- input width  (pixels)
  G_KERNEL    : positive := 3      -- kernel size (3 or 5)
  G_PADDING   : natural  := 1      -- zero-padding each side
  G_PAR_MACS  : positive           -- parallel MAC units (= parallel output channels)
  G_USE_BRAM  : boolean  := true   -- weights stored in BRAM (false = SDRAM-fed FIFO)
  G_USE_REGS  : boolean  := false  -- line buffers in registers instead of BRAM (use for small layers e.g. Conv1)

Ports:
  clk     : in  std_logic
  rst_n   : in  std_logic

  -- streaming input (from previous layer or top-level)
  i_valid : in  std_logic
  i_ready : out std_logic                       -- backpressure to upstream
  i_data  : in  std_logic_vector(7 downto 0)   -- uint8 activation (channel-last)
  i_last  : in  std_logic                       -- last byte of frame

  -- streaming output (to next layer)
  o_valid : out std_logic
  o_ready : in  std_logic                       -- backpressure from downstream
  o_data  : out std_logic_vector(7 downto 0)   -- uint8 activation (channel-last)
  o_last  : out std_logic                       -- last byte of frame
```

---

## Data format

Pixels arrive **channel-last**:

```
(row=0, col=0, ch=0), (row=0, col=0, ch=1), ..., (row=0, col=0, ch=C_IN-1),
(row=0, col=1, ch=0), ...
```

One byte per clock cycle when `i_valid = 1` and `i_ready = 1`.

---

## Internal structure

```
i_data
  │
  ▼
┌─────────────────┐
│   line_buffer   │  holds K rows (BRAM/registers)
│  + shift regs   │  holds K pixels per row tap (registers)
└────────┬────────┘
         │  K×K×C_IN window (valid signal)
         ▼
┌─────────────────┐
│    mac_array    │  G_PAR_MACS parallel MACs
│                 │  fires once per window position per output channel group
└────────┬────────┘
         │  G_PAR_MACS int32 accumulators
         ▼
┌─────────────────┐
│  requant_unit   │  G_PAR_MACS instances in parallel
│  (x G_PAR_MACS) │  clip( round( acc × m >> r ), 0, 255 )
└────────┬────────┘
         │  G_PAR_MACS uint8 values
         ▼
       o_data (streamed out one byte at a time)
```

Weights and biases live internally in `weight_bram` (G_WEIGHT_SRC="BRAM")
or are fed from `weight_fifo` filled by `sdram_ctrl` (G_WEIGHT_SRC="SDRAM").

---

## Internal FSM

The main counters drive stream position and output-channel grouping:

```
row_cnt : 0 → G_H_IN + G_PADDING - 1
col_cnt : 0 → G_W_IN + G_PADDING - 1
ch_cnt  : 0 → G_C_IN - 1
out_grp : 0 → G_C_OUT / G_PAR_MACS - 1
```

### States

**S_INIT**
- Accept initial real input pixels.
- Fill the current-row shift register and circular line buffer.
- Start computing as soon as padding makes a window valid:
  `row_cnt >= G_KERNEL - 1 - G_PADDING` and
  `col_cnt >= G_KERNEL - 1 - G_PADDING`.

**S_STEADY**
- Accept real input pixels while `row_cnt < G_H_IN` and `col_cnt < G_W_IN`.
- Hold `i_ready = 0` while generating right/bottom virtual zero-padding
  coordinates after the real image edge.
- Transition to `S_COMPUTE` for every valid output pixel coordinate.

**S_COMPUTE**
- Window is valid and all C_IN channels of the current real or virtual pixel
  coordinate are available.
- Pull `i_ready = 0` while MAC/output work is in progress.
- Iterate `out_grp` from 0 to G_C_OUT/G_PAR_MACS - 1:
  - Walk all `G_KERNEL * G_KERNEL * G_C_IN` MAC positions.
  - Accumulate into `G_PAR_MACS` parallel output channels.
  - Apply ReLU + requant.
  - Stream `G_PAR_MACS` bytes one byte at a time.
- Transition back to `S_STEADY` after all output channel groups for the
  current output pixel have been emitted.

**S_FLUSH**
- Reached after the final padded output coordinate has been emitted.
- Reset `row_cnt`, `col_cnt`, `ch_cnt`, `out_grp` to 0
- Transition → `S_INIT`

---

## Timing

One output pixel (all G_C_OUT channels) takes:

```
G_C_OUT / G_PAR_MACS  cycles  (one cycle per output group)
```

Input is stalled during this time via backpressure (`i_ready = 0`).

For Conv1 (G_C_OUT=64, G_PAR_MACS=2):  32 stall cycles per output pixel
For Conv4 (G_C_OUT=256, G_PAR_MACS=73): 4 stall cycles per output pixel

---

## Notes

- ReLU is applied as `clamp(acc, min=0)` on the int32 accumulator before requant —
  it is not a separate module
- Top/left zero-padding is handled by treating out-of-bounds window positions
  as zero during MAC reads.
- Right/bottom zero-padding is generated internally after the real image edge
  while holding `i_ready = 0`; upstream does not send explicit pad bytes.
- Activations are consumed as uint8 values (`0..255`) and multiplied by int8
  weights. The MAC path zero-extends activation bytes to 9-bit signed before
  multiplying so values above 127 are not interpreted as negative.
- The line buffer has `G_KERNEL` rows, not `G_KERNEL - 1`, because the current
  input row is written immediately and must not overwrite the oldest row still
  needed by a padded `K x K` window.
- G_PAR_MACS must divide G_C_OUT exactly
- The same conv_layer entity is used for all five conv layers — only the generic
  map changes at the top level
- G_USE_REGS=true is only practical for small line buffers.
  For larger layers (Conv2-5) always use G_USE_REGS=false (BRAM).
  The compute logic is identical either way — only the line buffer storage changes.

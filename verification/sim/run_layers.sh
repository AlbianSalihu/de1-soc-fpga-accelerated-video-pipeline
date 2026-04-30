#!/usr/bin/env bash
# Run all layer-level testbenches.
# Must be invoked from the project root:
#   bash verification/sim/run_layers.sh

set -euo pipefail

WORKDIR=verification/sim
STD="--std=08"
WORK="--work=work --workdir=$WORKDIR"
VECTORS_DIR=${VECTORS_DIR:-verification/vectors/default}
RESULTS_DIR=${RESULTS_DIR:-verification/results/default}
FPGAQPARMS_JSON=${FPGAQPARMS_JSON:-ml/outputs/run0/fpgaqparms.json}
PAR_MACS_DEFAULT=${PAR_MACS_DEFAULT:-64}
PROGRESS_STEP=${PROGRESS_STEP:-10}

with_trailing_slash() {
    case "$1" in
        */) printf "%s" "$1" ;;
        *)  printf "%s/" "$1" ;;
    esac
}

VECTORS_G=$(with_trailing_slash "$VECTORS_DIR")
RESULTS_G=$(with_trailing_slash "$RESULTS_DIR")

mkdir -p "$RESULTS_DIR"

pass=0
fail=0
ran=0

selected=("$@")

is_selected() {
    local prefix="$1"

    if [ ${#selected[@]} -eq 0 ]; then
        return 0
    fi

    for item in "${selected[@]}"; do
        if [ "$item" = "$prefix" ]; then
            return 0
        fi
    done

    return 1
}

analyse_tb() {
    local tb_name="$1"
    local sources=("${@:2}")

    echo "── $tb_name ──────────────────────────────"

    # analyse all source files
    for src in "${sources[@]}"; do
        ghdl -a $STD $WORK "$src"
    done

    # elaborate
    ghdl -e $STD $WORK "$tb_name"
}

run_conv() {
    local prefix="$1"
    local c_in="$2"
    local c_out="$3"
    local h_in="$4"
    local w_in="$5"
    local kernel="$6"
    local padding="$7"
    local par_macs="$8"

    if ! is_selected "$prefix"; then
        return
    fi

    ran=$((ran + 1))

    echo "── tb_conv_layer: $prefix ──────────────────────────────"
    printf "   shape: Cin=%s Cout=%s HxW=%sx%s K=%s P=%s PAR=%s\n" \
        "$c_in" "$c_out" "$h_in" "$w_in" "$kernel" "$padding" "$par_macs"

    # run — GHDL exits 0 on stop(0), non-zero on failure
    if ghdl -r $STD $WORK tb_conv_layer \
        -gG_PREFIX="$prefix" \
        -gG_C_IN="$c_in" \
        -gG_C_OUT="$c_out" \
        -gG_H_IN="$h_in" \
        -gG_W_IN="$w_in" \
        -gG_KERNEL="$kernel" \
        -gG_PADDING="$padding" \
        -gG_PAR_MACS="$par_macs" \
        -gG_VECS="$VECTORS_G" \
        -gG_RESS="$RESULTS_G" \
        -gG_PROGRESS_STEP="$PROGRESS_STEP"; then
        echo "PASS: $prefix"
        pass=$((pass + 1))
    else
        echo "FAIL: $prefix"
        fail=$((fail + 1))
    fi

    echo ""
}

# ── conv_layer ────────────────────────────────────────────────────────────────
analyse_tb tb_conv_layer \
    hardware/rtl/layers/conv_layer.vhd \
    verification/tb/layers/tb_conv_layer.vhd

while read -r prefix c_in c_out h_in w_in kernel padding par_macs; do
    run_conv "$prefix" "$c_in" "$c_out" "$h_in" "$w_in" "$kernel" "$padding" "$par_macs"
done < <(
    python - "$VECTORS_DIR" "$FPGAQPARMS_JSON" "$PAR_MACS_DEFAULT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch.nn as nn

from ml.src.models.alexnet64gray import AlexNet64Gray

vec_dir = Path(sys.argv[1])
meta_path = Path(sys.argv[2])
par_default = int(sys.argv[3])

def die(msg: str) -> None:
    raise SystemExit(f"run_layers.sh: {msg}")

def require_file(prefix: str, suffix: str) -> None:
    path = vec_dir / f"{prefix}_{suffix}.bin"
    if not path.exists():
        die(f"missing {path}")

def choose_par(c_out: int) -> int:
    limit = min(par_default, c_out)
    for candidate in range(limit, 0, -1):
        if c_out % candidate == 0:
            return candidate
    return 1

def pair(value, label: str) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            die(f"unsupported {label}: {value}")
        return int(value[0]), int(value[1])
    return int(value), int(value)

def conv_out_size(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return ((size + 2 * padding - dilation * (kernel - 1) - 1) // stride) + 1

if not meta_path.exists():
    die(f"missing metadata JSON: {meta_path}")

meta = json.loads(meta_path.read_text())
layer_meta = {entry["name"]: entry for entry in meta["layers"]}

model = AlexNet64Gray()
h = int(meta.get("mnist64_cfg", {}).get("image_size", 64))
w = h

for child_name, mod in model.features.named_children():
    lname = f"features.{child_name}"

    if isinstance(mod, nn.Conv2d):
        if lname not in layer_meta:
            die(f"{lname} is in the model but missing from {meta_path}")

        entry = layer_meta[lname]
        if entry.get("type") != "Conv2d":
            die(f"{lname}: metadata type is {entry.get('type')}, expected Conv2d")

        c_out, c_in, kh_meta, kw_meta = [int(v) for v in entry["weight_shape"]]
        kh, kw = pair(mod.kernel_size, f"{lname} kernel_size")
        sh, sw = pair(mod.stride, f"{lname} stride")
        ph, pw = pair(mod.padding, f"{lname} padding")
        dh, dw = pair(mod.dilation, f"{lname} dilation")

        if kh != kw:
            die(f"{lname}: non-square kernels are not supported by conv_layer")
        if ph != pw:
            die(f"{lname}: asymmetric padding is not supported by conv_layer")
        if sh != 1 or sw != 1:
            die(f"{lname}: conv_layer testbench only supports stride=1")
        if dh != 1 or dw != 1:
            die(f"{lname}: conv_layer testbench only supports dilation=1")
        if kh != kh_meta or kw != kw_meta:
            die(f"{lname}: model kernel {kh}x{kw} does not match metadata {kh_meta}x{kw_meta}")

        h_out = conv_out_size(h, kh, sh, ph, dh)
        w_out = conv_out_size(w, kw, sw, pw, dw)
        if h_out != h or w_out != w:
            die(f"{lname}: conv_layer expects same-size output, got {h_out}x{w_out} from {h}x{w}")

        prefix = lname.replace(".", "_")
        for suffix in ("in", "out", "weights", "biases", "requant_m", "requant_r"):
            require_file(prefix, suffix)

        print(prefix, c_in, c_out, h, w, kh, ph, choose_par(c_out))

        h, w = h_out, w_out

    elif isinstance(mod, nn.MaxPool2d):
        kh, kw = pair(mod.kernel_size, f"{lname} kernel_size")
        sh, sw = pair(mod.stride, f"{lname} stride")
        ph, pw = pair(mod.padding, f"{lname} padding")
        dh, dw = pair(mod.dilation, f"{lname} dilation")
        h = conv_out_size(h, kh, sh, ph, dh)
        w = conv_out_size(w, kw, sw, pw, dw)

    elif isinstance(mod, nn.ReLU):
        pass

    else:
        die(f"unsupported feature module {lname}: {mod.__class__.__name__}")
PY
)

# ── summary ──────────────────────────────────────────────────────────────────
if [ "$ran" -eq 0 ]; then
    echo "No selected layers matched: ${selected[*]}"
    exit 1
fi

echo "Results: $pass passed, $fail failed"
[ $fail -eq 0 ]

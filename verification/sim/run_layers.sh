#!/usr/bin/env bash
#
# Run the current streaming conv_layer against the existing model vectors.
#
# Must be invoked from the project root:
#
#   bash verification/sim/run_layers.sh
#
# Run selected layers:
#
#   bash verification/sim/run_layers.sh features_0
#
# Run only the first output group:
#
#   MAX_GROUPS=1 bash verification/sim/run_layers.sh features_0
#
# Force regeneration of raw accumulator golden files:
#
#   FORCE_GOLDEN=1 bash verification/sim/run_layers.sh
#
# Generate GHW waveforms:
#
#   WAVE=1 MAX_GROUPS=1 \
#       bash verification/sim/run_layers.sh features_0
#

set -euo pipefail


WORKDIR=${WORKDIR:-verification/sim/work_layers}

STD=(--std=08)
WORK=(--work=work --workdir="$WORKDIR")

VECTORS_DIR=${VECTORS_DIR:-verification/vectors/default}
RESULTS_DIR=${RESULTS_DIR:-verification/results/default}
RAW_ACC_DIR=${RAW_ACC_DIR:-"$RESULTS_DIR/raw_acc"}

FPGAQPARMS_JSON=${FPGAQPARMS_JSON:-ml/outputs/run0/fpgaqparms.json}

PAR_MACS_DEFAULT=${PAR_MACS_DEFAULT:-64}
MAX_GROUPS=${MAX_GROUPS:-0}

STALL_PERIOD=${STALL_PERIOD:-7}
PROGRESS_STEP=${PROGRESS_STEP:-10}
TIMEOUT_CYCLES=${TIMEOUT_CYCLES:-0}

FORCE_GOLDEN=${FORCE_GOLDEN:-0}
WAVE=${WAVE:-0}


with_trailing_slash() {
    case "$1" in
        */) printf "%s" "$1" ;;
        *)  printf "%s/" "$1" ;;
    esac
}


VECTORS_G=$(with_trailing_slash "$VECTORS_DIR")

mkdir -p "$WORKDIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$RAW_ACC_DIR"

selected=("$@")


for required_file in \
    hardware/rtl/layers/conv_layer.vhd \
    verification/tb/layers/tb_conv_layer.vhd
do
    if [ ! -f "$required_file" ]; then
        echo "run_layers.sh: missing $required_file" >&2
        exit 1
    fi
done


if [ ! -f "$FPGAQPARMS_JSON" ]; then
    echo \
        "run_layers.sh: missing metadata JSON: $FPGAQPARMS_JSON" \
        >&2
    exit 1
fi


MANIFEST=$(mktemp "$WORKDIR/conv_layers_manifest.XXXXXX")
trap 'rm -f "$MANIFEST"' EXIT


echo "Preparing layer metadata and raw accumulator golden files..."

python - \
    "$VECTORS_DIR" \
    "$FPGAQPARMS_JSON" \
    "$PAR_MACS_DEFAULT" \
    "$RAW_ACC_DIR" \
    "$FORCE_GOLDEN" \
    "${selected[@]}" \
    > "$MANIFEST" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch.nn as nn

from ml.src.models.alexnet64gray import AlexNet64Gray


vectors_dir = Path(sys.argv[1])
metadata_path = Path(sys.argv[2])
par_default = int(sys.argv[3])
raw_acc_dir = Path(sys.argv[4])
force_golden = bool(int(sys.argv[5]))
selected = set(sys.argv[6:])


def die(message: str) -> None:
    raise SystemExit(f"run_layers.sh: {message}")


def pair(value, label: str) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            die(f"unsupported {label}: {value}")
        return int(value[0]), int(value[1])

    return int(value), int(value)


def conv_out_size(
    size: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    return (
        (
            size
            + 2 * padding
            - dilation * (kernel - 1)
            - 1
        )
        // stride
    ) + 1


def choose_parallelism(c_out: int) -> int:
    limit = min(par_default, c_out)

    for candidate in range(limit, 0, -1):
        if c_out % candidate == 0:
            return candidate

    return 1


def require_file(path: Path) -> None:
    if not path.is_file():
        die(f"missing vector file: {path}")


def generate_raw_accumulators(
    *,
    prefix: str,
    c_in: int,
    c_out: int,
    h_in: int,
    w_in: int,
    kernel: int,
    input_path: Path,
    weight_path: Path,
    output_path: Path,
) -> None:
    h_out = h_in - kernel + 1
    w_out = w_in - kernel + 1

    if h_out <= 0 or w_out <= 0:
        die(
            f"{prefix}: invalid valid-convolution output "
            f"{h_out}x{w_out}"
        )

    activations_flat = np.fromfile(
        input_path,
        dtype=np.uint8,
    )

    expected_input_count = h_in * w_in * c_in

    if activations_flat.size != expected_input_count:
        die(
            f"{input_path}: expected "
            f"{expected_input_count} bytes, got "
            f"{activations_flat.size}"
        )

    weights_flat = np.fromfile(
        weight_path,
        dtype=np.int8,
    )

    expected_weight_count = (
        c_out * c_in * kernel * kernel
    )

    if weights_flat.size != expected_weight_count:
        die(
            f"{weight_path}: expected "
            f"{expected_weight_count} bytes, got "
            f"{weights_flat.size}"
        )

    activations = activations_flat.reshape(
        h_in,
        w_in,
        c_in,
    ).astype(np.int64)

    weights = weights_flat.reshape(
        c_out,
        c_in,
        kernel,
        kernel,
    ).astype(np.int64)

    accumulators = np.zeros(
        (h_out, w_out, c_out),
        dtype=np.int64,
    )

    # Golden valid convolution:
    #
    # output[row, col, c_out] =
    #     sum over c_in, kh, kw of
    #     input[row + kh, col + kw, c_in]
    #     * weight[c_out, c_in, kh, kw]
    #
    # The tensordot contracts only the input-channel dimension.
    for kernel_row in range(kernel):
        for kernel_column in range(kernel):
            activation_patch = activations[
                kernel_row : kernel_row + h_out,
                kernel_column : kernel_column + w_out,
                :,
            ]

            weight_slice = weights[
                :,
                :,
                kernel_row,
                kernel_column,
            ]

            accumulators += np.tensordot(
                activation_patch,
                weight_slice,
                axes=([2], [1]),
            )

    # The RTL uses signed 32-bit accumulators. Conversion to int32
    # reproduces two's-complement wrapping if a test ever overflows.
    accumulators.astype("<i4").tofile(output_path)


if not metadata_path.is_file():
    die(f"missing metadata JSON: {metadata_path}")

raw_acc_dir.mkdir(parents=True, exist_ok=True)

metadata = json.loads(metadata_path.read_text())

if "layers" not in metadata:
    die(f"{metadata_path} does not contain a layers array")

layer_metadata = {
    entry["name"]: entry
    for entry in metadata["layers"]
}

model = AlexNet64Gray()

h = int(
    metadata.get(
        "mnist64_cfg",
        {},
    ).get(
        "image_size",
        64,
    )
)

w = h

matched: set[str] = set()


for child_name, module in model.features.named_children():
    layer_name = f"features.{child_name}"
    prefix = layer_name.replace(".", "_")

    if isinstance(module, nn.Conv2d):
        if layer_name not in layer_metadata:
            die(
                f"{layer_name} exists in the model but is "
                f"missing from {metadata_path}"
            )

        entry = layer_metadata[layer_name]

        if entry.get("type") != "Conv2d":
            die(
                f"{layer_name}: metadata type is "
                f"{entry.get('type')}, expected Conv2d"
            )

        c_out, c_in, kh_metadata, kw_metadata = [
            int(value)
            for value in entry["weight_shape"]
        ]

        kh, kw = pair(
            module.kernel_size,
            f"{layer_name} kernel_size",
        )

        sh, sw = pair(
            module.stride,
            f"{layer_name} stride",
        )

        ph, pw = pair(
            module.padding,
            f"{layer_name} padding",
        )

        dh, dw = pair(
            module.dilation,
            f"{layer_name} dilation",
        )

        if kh != kw:
            die(
                f"{layer_name}: non-square kernels are "
                f"not supported"
            )

        if kh != kh_metadata or kw != kw_metadata:
            die(
                f"{layer_name}: model kernel "
                f"{kh}x{kw} does not match metadata "
                f"{kh_metadata}x{kw_metadata}"
            )

        if sh != 1 or sw != 1:
            die(
                f"{layer_name}: current conv_layer "
                f"verification requires stride=1"
            )

        if dh != 1 or dw != 1:
            die(
                f"{layer_name}: current conv_layer "
                f"verification requires dilation=1"
            )

        # The existing input vector belongs to the padded model layer,
        # but the current RTL computes only the valid spatial region.
        valid_h_out = h - kh + 1
        valid_w_out = w - kw + 1

        if valid_h_out <= 0 or valid_w_out <= 0:
            die(
                f"{layer_name}: input {h}x{w} is too small "
                f"for kernel {kh}"
            )

        input_path = vectors_dir / f"{prefix}_in.bin"
        weight_path = vectors_dir / f"{prefix}_weights.bin"
        raw_path = raw_acc_dir / f"{prefix}_raw_acc.bin"

        require_file(input_path)
        require_file(weight_path)

        if not selected or prefix in selected:
            matched.add(prefix)

            if force_golden or not raw_path.is_file():
                print(
                    f"Generating {raw_path}",
                    file=sys.stderr,
                )

                generate_raw_accumulators(
                    prefix=prefix,
                    c_in=c_in,
                    c_out=c_out,
                    h_in=h,
                    w_in=w,
                    kernel=kh,
                    input_path=input_path,
                    weight_path=weight_path,
                    output_path=raw_path,
                )

            expected_raw_bytes = (
                valid_h_out
                * valid_w_out
                * c_out
                * 4
            )

            actual_raw_bytes = raw_path.stat().st_size

            if actual_raw_bytes != expected_raw_bytes:
                die(
                    f"{raw_path}: expected "
                    f"{expected_raw_bytes} bytes, got "
                    f"{actual_raw_bytes}. "
                    f"Use FORCE_GOLDEN=1 to regenerate it."
                )

            parallelism = choose_parallelism(c_out)

            print(
                prefix,
                c_in,
                c_out,
                h,
                w,
                kh,
                parallelism,
                raw_path,
            )

        # Track the real model feature-map dimensions for the input
        # vector of the next layer.
        h = conv_out_size(
            h,
            kh,
            sh,
            ph,
            dh,
        )

        w = conv_out_size(
            w,
            kw,
            sw,
            pw,
            dw,
        )

    elif isinstance(module, nn.MaxPool2d):
        kh, kw = pair(
            module.kernel_size,
            f"{layer_name} kernel_size",
        )

        sh, sw = pair(
            module.stride,
            f"{layer_name} stride",
        )

        ph, pw = pair(
            module.padding,
            f"{layer_name} padding",
        )

        dh, dw = pair(
            module.dilation,
            f"{layer_name} dilation",
        )

        h = conv_out_size(
            h,
            kh,
            sh,
            ph,
            dh,
        )

        w = conv_out_size(
            w,
            kw,
            sw,
            pw,
            dw,
        )

    elif isinstance(module, nn.ReLU):
        pass

    else:
        die(
            f"unsupported feature module {layer_name}: "
            f"{module.__class__.__name__}"
        )


if selected:
    missing = selected - matched

    if missing:
        die(
            "selected layer prefixes were not found: "
            + ", ".join(sorted(missing))
        )
PY


if [ ! -s "$MANIFEST" ]; then
    if [ ${#selected[@]} -eq 0 ]; then
        echo "run_layers.sh: no convolution layers were discovered" >&2
    else
        echo \
            "run_layers.sh: no selected layers matched: ${selected[*]}" \
            >&2
    fi

    exit 1
fi


echo ""
echo "Analyzing conv_layer..."

ghdl -a \
    "${STD[@]}" \
    "${WORK[@]}" \
    hardware/rtl/layers/conv_layer.vhd


echo "Analyzing vector-driven testbench..."

ghdl -a \
    "${STD[@]}" \
    "${WORK[@]}" \
    verification/tb/layers/tb_conv_layer.vhd


echo "Elaborating tb_conv_layer..."

ghdl -e \
    "${STD[@]}" \
    "${WORK[@]}" \
    tb_conv_layer


pass=0
fail=0
ran=0


while read -r \
    prefix \
    c_in \
    c_out \
    h_in \
    w_in \
    kernel \
    c_par \
    raw_file
do
        total_groups=$((c_out / c_par))
    groups_to_run=$total_groups

    h_out=$((h_in - kernel + 1))
    w_out=$((w_in - kernel + 1))
    window_count=$((h_out * w_out))
    kernel_size=$((kernel * kernel))

    activation_cycles=$((h_in * w_in * c_in))
    mac_cycles=$((window_count * c_in * kernel_size))

    if [ "$c_in" -eq 1 ]; then
        # Single-channel layers load the weight group once.
        weight_cycles=$((c_par * kernel_size))
    else
        # Multi-channel layers reload one C_PAR × K² weight group
        # for every input channel of every output window.
        weight_cycles=$(
            (
                window_count *
                c_in *
                c_par *
                kernel_size
            )
        )
    fi

    # Extra time for FSM transitions, output backpressure,
    # line filling and testbench file handling.
    estimated_cycles=$(
        (
            activation_cycles +
            mac_cycles +
            weight_cycles +
            window_count * 32 +
            10000
        )
    )

    if [ "$TIMEOUT_CYCLES" -gt 0 ]; then
        group_timeout_cycles=$TIMEOUT_CYCLES
    else
        group_timeout_cycles=$((estimated_cycles * 2))
    fi

    if [ "$MAX_GROUPS" -gt 0 ] &&
       [ "$MAX_GROUPS" -lt "$groups_to_run" ]; then

        groups_to_run=$MAX_GROUPS
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Layer: $prefix"
    echo \
        "Shape: Cin=$c_in Cout=$c_out HxW=${h_in}x${w_in} " \
        "K=$kernel Cpar=$c_par"
    echo \
        "Valid output: $((h_in - kernel + 1))x" \
        "$((w_in - kernel + 1))"
    echo \
        "Groups: $groups_to_run/$total_groups"
    echo \
        "Watchdog: $group_timeout_cycles cycles " \
        "($((group_timeout_cycles * 10)) ns)"
        
    echo "Golden: $raw_file"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for ((group = 0; group < groups_to_run; group++)); do
        ran=$((ran + 1))

        first_channel=$((group * c_par))
        last_channel=$((first_channel + c_par - 1))

        result_file="$RESULTS_DIR/"\
"${prefix}_group_${group}_received_acc.bin"

        wave_file="$RESULTS_DIR/"\
"${prefix}_group_${group}.ghw"

        echo ""
        echo \
            "── $prefix group $group " \
            "channels $first_channel..$last_channel ──"

        run_command=(
            ghdl -r
            "${STD[@]}"
            "${WORK[@]}"
            tb_conv_layer

            "-gG_PREFIX=$prefix"

            "-gG_C_IN=$c_in"
            "-gG_C_OUT=$c_out"
            "-gG_H_IN=$h_in"
            "-gG_W_IN=$w_in"
            "-gG_KERNEL=$kernel"
            "-gG_C_PAR=$c_par"

            "-gG_OUT_GROUP=$group"

            "-gG_VECS=$VECTORS_G"
            "-gG_GOLDEN_FILE=$raw_file"
            "-gG_RESULT_FILE=$result_file"

            "-gG_STALL_PERIOD=$STALL_PERIOD"
            "-gG_PROGRESS_STEP=$PROGRESS_STEP"
            "-gG_TIMEOUT_CYCLES=$group_timeout_cycles"

            --assert-level=error
        )

        if [ "$WAVE" -eq 1 ]; then
            run_command+=("--wave=$wave_file")
        fi

        if "${run_command[@]}"; then
            echo \
                "PASS: $prefix group $group " \
                "channels $first_channel..$last_channel"

            pass=$((pass + 1))
        else
            echo \
                "FAIL: $prefix group $group " \
                "channels $first_channel..$last_channel"

            fail=$((fail + 1))
        fi
    done
done < "$MANIFEST"


echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Convolution vector results"
echo "  Ran:    $ran groups"
echo "  Passed: $pass"
echo "  Failed: $fail"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


if [ "$ran" -eq 0 ]; then
    exit 1
fi

[ "$fail" -eq 0 ]
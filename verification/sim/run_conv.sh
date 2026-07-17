#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR="verification/sim/build/conv"
RESULT_DIR="verification/results/directed/conv"

RTL_FILE="hardware/rtl/layers/conv_layer.vhd"
TB_FILE="verification/tb/conv/tb_conv_layer_initial_line_fill.vhd"
TB_ENTITY="tb_conv_layer_initial_line_fill"

mkdir -p "$BUILD_DIR"
mkdir -p "$RESULT_DIR"

echo "Analyzing conv_layer..."
ghdl -a \
    --std=08 \
    --workdir="$BUILD_DIR" \
    "$RTL_FILE"

echo "Analyzing $TB_ENTITY..."
ghdl -a \
    --std=08 \
    --workdir="$BUILD_DIR" \
    "$TB_FILE"

echo "Elaborating $TB_ENTITY..."
ghdl -e \
    --std=08 \
    --workdir="$BUILD_DIR" \
    "$TB_ENTITY"

echo "Running $TB_ENTITY..."
ghdl -r \
    --std=08 \
    --workdir="$BUILD_DIR" \
    "$TB_ENTITY" \
    --assert-level=error \
    --wave="$RESULT_DIR/tb_conv_layer_initial_line_fill.ghw"

echo "Simulation completed successfully."
echo "Waveform: $RESULT_DIR/tb_conv_layer_initial_line_fill.ghw"
#!/usr/bin/env bash
set -e  # stop au premier crash

# ==============================
# Configuration
# ==============================
CAW_DIR="Catch-A-Waveform"
INPUT_DIR="${CAW_DIR}/inputs"
FILE_NAME="matuidi_charo.wav"


# ==============================
# Entraînement
# ==============================
echo "🚀 Starting Catch-A-Waveform training"
echo "🎵 Input file: $FILE_NAME"

cd "$CAW_DIR"

python train_main.py \
  --input_file "$FILE_NAME" \

echo "✅ Training finished"

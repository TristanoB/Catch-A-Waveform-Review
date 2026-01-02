#!/usr/bin/env bash
set -e  # stop au premier crash

# ==============================
# Configuration
# ==============================
CAW_DIR="Catch-A-Waveform"
INPUT_DIR="${CAW_DIR}/inputs"
FILE_NAME="oiseaux.wav"

# ==============================
# Vérifications
# ==============================
echo "🔍 Checking Catch-A-Waveform directory..."
if [ ! -d "$CAW_DIR" ]; then
  echo "❌ Catch-A-Waveform directory not found: $CAW_DIR"
  exit 1
fi

echo "🔍 Checking input audio..."
if [ ! -f "$INPUT_DIR/$FILE_NAME" ]; then
  echo "❌ Audio file not found: $INPUT_DIR/$FILE_NAME"
  echo "👉 Expected structure:"
  echo "   Catch-A-Waveform/inputs/$FILE_NAME"
  exit 1
fi

# ==============================
# Entraînement
# ==============================
echo "🚀 Starting Catch-A-Waveform training"
echo "🎵 Input file: $FILE_NAME"

cd "$CAW_DIR"

python train_main.py \
  --input_file "$FILE_NAME"

echo "✅ Training finished"

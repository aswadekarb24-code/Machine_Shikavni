#!/bin/bash
# Script to download and extract traffic sign datasets into data/raw_data/
set -e

RAW_DIR="data/raw_data"

echo "------------------------------------------------"
echo "Setting up Python virtual environment..."

python -m venv .venv

echo "Installing Python dependencies..."
source .venv/bin/activate
pip install -r requirements.txt

echo "------------------------------------------------"
echo "Creating raw data directories..."
mkdir -p "$RAW_DIR/CTSD"
mkdir -p "$RAW_DIR/GTSRB"
mkdir -p "$RAW_DIR/BTSD"

echo "------------------------------------------------"
echo "[1/3] Downloading CTSD (Chinese Traffic Signs)..."
curl -L -o "$RAW_DIR/CTSD/chinese-traffic-signs.zip" \
  https://www.kaggle.com/api/v1/datasets/download/dmitryyemelyanov/chinese-traffic-signs

echo "Extracting CTSD..."
unzip -q -o "$RAW_DIR/CTSD/chinese-traffic-signs.zip" -d "$RAW_DIR/CTSD/"
rm "$RAW_DIR/CTSD/chinese-traffic-signs.zip"

echo "------------------------------------------------"
echo "[2/3] Downloading GTSRB (German Traffic Signs)..."
curl -L -o "$RAW_DIR/GTSRB/gtsrb-german-traffic-sign.zip" \
  https://www.kaggle.com/api/v1/datasets/download/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

echo "Extracting GTSRB..."
unzip -q -o "$RAW_DIR/GTSRB/gtsrb-german-traffic-sign.zip" -d "$RAW_DIR/GTSRB/"
rm "$RAW_DIR/GTSRB/gtsrb-german-traffic-sign.zip"

echo "------------------------------------------------"
echo "[3/3] Downloading BTSD (Belgium Traffic Signs)..."
curl -L -o "$RAW_DIR/BTSD/belgium-ts.zip" \
  https://www.kaggle.com/api/v1/datasets/download/abhi8923shriv/belgium-ts

echo "Extracting BTSD..."
unzip -q -o "$RAW_DIR/BTSD/belgium-ts.zip" -d "$RAW_DIR/BTSD/"
rm "$RAW_DIR/BTSD/belgium-ts.zip"

echo "------------------------------------------------"
echo "All datasets downloaded and extracted successfully into $RAW_DIR!"

echo "------------------------------------------------"
echo "Preprocessing Data..."
python -m src.preproc.crop

echo "------------------------------------------------"
echo "Running Dataloaders..."
python -m src.preproc.factory

echo "------------------------------------------------"
echo "Setup and preprocessing completed successfully!"
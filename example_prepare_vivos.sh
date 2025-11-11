#!/bin/bash
# Script ví dụ để chuẩn bị dữ liệu VIVOS cho fine-tuning

echo "=== Chuẩn bị dữ liệu VIVOS cho fine-tuning ==="

# Tạo thư mục dataset
mkdir -p dataset

# Copy audio files (hoặc tạo symlink)
echo "Đang copy audio files..."
# Option 1: Copy (tốn dung lượng)
# cp -r archive/vivos/train/waves/* dataset/

# Option 2: Tạo symlink (tiết kiệm dung lượng)
# Windows: mklink /D dataset archive\vivos\train\waves
# Linux/Mac: ln -s archive/vivos/train/waves dataset

# Tạo train.jsonl
echo "Đang tạo train.jsonl..."
python prepare_jsonl.py \
    --audio-dir "archive/vivos/train/waves" \
    --prompts-file "archive/vivos/train/prompts.txt" \
    --output "train.jsonl" \
    --dataset-name "dataset"

# Tạo test.jsonl (nếu có)
echo "Đang tạo test.jsonl..."
python prepare_jsonl.py \
    --audio-dir "archive/vivos/test/waves" \
    --prompts-file "archive/vivos/test/prompts.txt" \
    --output "test.jsonl" \
    --dataset-name "dataset"

# Validate datasets
echo "Đang validate train.jsonl..."
python validate_dataset.py \
    --jsonl-file "train.jsonl" \
    --audio-dir "archive/vivos/train/waves"

echo "Đang validate test.jsonl..."
python validate_dataset.py \
    --jsonl-file "test.jsonl" \
    --audio-dir "archive/vivos/test/waves"

echo "=== Hoàn thành! ==="



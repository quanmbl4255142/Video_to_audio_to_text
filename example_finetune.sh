#!/bin/bash
# Script ví dụ để fine-tune Whisper/PhoWhisper

echo "=== Fine-tuning Whisper/PhoWhisper ==="

# Fine-tune với PhoWhisper-base (khuyến nghị cho tiếng Việt)
python fine_tune_whisper.py \
    --model-name "vinai/PhoWhisper-base" \
    --train-jsonl "train.jsonl" \
    --audio-dir "archive/vivos/train/waves" \
    --eval-jsonl "test.jsonl" \
    --output-dir "./whisper-finetuned-phowhisper-base" \
    --num-epochs 3 \
    --batch-size 16 \
    --learning-rate 1e-5 \
    --warmup-steps 500 \
    --fp16

echo "=== Hoàn thành fine-tuning! ==="
echo "Model đã được lưu tại: ./whisper-finetuned-phowhisper-base"



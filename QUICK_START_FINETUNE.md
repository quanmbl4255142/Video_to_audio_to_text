# 🚀 Quick Start - Fine-tuning Whisper/PhoWhisper

Hướng dẫn nhanh để bắt đầu fine-tuning trong 5 phút.

## Bước 1: Cài đặt

```bash
pip install -r requirements.txt
```

## Bước 2: Chuẩn bị dữ liệu

### Từ VIVOS dataset:

```bash
# Tạo train.jsonl
python prepare_jsonl.py \
    --audio-dir "archive/vivos/train/waves" \
    --prompts-file "archive/vivos/train/prompts.txt" \
    --output "train.jsonl" \
    --dataset-name "dataset"

# Tạo test.jsonl (optional)
python prepare_jsonl.py \
    --audio-dir "archive/vivos/test/waves" \
    --prompts-file "archive/vivos/test/prompts.txt" \
    --output "test.jsonl" \
    --dataset-name "dataset"
```

### Validate dữ liệu:

```bash
python validate_dataset.py \
    --jsonl-file "train.jsonl" \
    --audio-dir "archive/vivos/train/waves"
```

## Bước 3: Fine-tuning

### Với GPU (khuyến nghị):

```bash
python fine_tune_whisper.py \
    --model-name "vinai/PhoWhisper-base" \
    --train-jsonl "train.jsonl" \
    --audio-dir "archive/vivos/train/waves" \
    --eval-jsonl "test.jsonl" \
    --output-dir "./whisper-finetuned" \
    --num-epochs 3 \
    --batch-size 16 \
    --fp16
```

### Với CPU (chậm hơn):

```bash
python fine_tune_whisper.py \
    --model-name "vinai/PhoWhisper-base" \
    --train-jsonl "train.jsonl" \
    --audio-dir "archive/vivos/train/waves" \
    --output-dir "./whisper-finetuned" \
    --num-epochs 2 \
    --batch-size 4
```

## Bước 4: Sử dụng model đã fine-tune

```python
from transformers import pipeline
import torch

pipe = pipeline(
    "automatic-speech-recognition",
    model="./whisper-finetuned",
    device=0 if torch.cuda.is_available() else -1,
)

result = pipe("path/to/audio.wav")
print(result["text"])
```

## Format JSONL

Mỗi dòng trong file JSONL:

```json
{"audio": "dataset/ad001.wav", "sentence": "Sắm Tết cùng Shopee nhận ngay quà hấp dẫn!"}
{"audio": "dataset/ad002.wav", "sentence": "Dầu gội Head & Shoulders đánh bay gàu tức thì."}
```

## Models có sẵn

- `vinai/PhoWhisper-base` - **Khuyến nghị cho tiếng Việt**
- `vinai/PhoWhisper-large` - Chính xác hơn, lớn hơn
- `openai/whisper-base` - Whisper gốc
- `openai/whisper-small` - Whisper nhỏ hơn

## Troubleshooting

**Out of Memory?**
- Giảm `--batch-size` (16 → 8 → 4)
- Bật `--fp16` nếu có GPU
- Sử dụng model nhỏ hơn

**Training quá chậm?**
- Sử dụng GPU
- Tăng `--batch-size` nếu đủ VRAM
- Giảm số epochs

Xem `FINE_TUNING_GUIDE.md` để biết chi tiết đầy đủ!



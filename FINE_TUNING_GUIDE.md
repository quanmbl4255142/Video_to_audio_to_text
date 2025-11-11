# 🎯 Hướng Dẫn Fine-tuning Whisper/PhoWhisper

Hướng dẫn chi tiết để fine-tune model Whisper hoặc PhoWhisper với dữ liệu tiếng Việt của bạn.

## 📋 Mục Lục

1. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
2. [Cài Đặt](#cài-đặt)
3. [Chuẩn Bị Dữ Liệu](#chuẩn-bị-dữ-liệu)
4. [Validate Dữ Liệu](#validate-dữ-liệu)
5. [Fine-tuning](#fine-tuning)
6. [Sử Dụng Model Đã Fine-tune](#sử-dụng-model-đã-fine-tune)
7. [Troubleshooting](#troubleshooting)

---

## 🖥️ Yêu Cầu Hệ Thống

### Tối Thiểu:
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 10GB trống
- **GPU**: Không bắt buộc nhưng khuyến nghị (NVIDIA GPU với 6GB+ VRAM)

### Khuyến Nghị:
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 50GB+ trống (cho model và cache)
- **GPU**: NVIDIA GPU với CUDA support (8GB+ VRAM)

---

## 📦 Cài Đặt

### 1. Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

### 2. Cài đặt thêm cho GPU (nếu có NVIDIA GPU):

```bash
# Kiểm tra CUDA version
nvidia-smi

# Cài PyTorch với CUDA (thay đổi cu118 theo CUDA version của bạn)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📁 Chuẩn Bị Dữ Liệu

### Format JSONL Chuẩn

Dữ liệu training phải ở format JSONL với cấu trúc:

```json
{"audio": "dataset/ad001.wav", "sentence": "Sắm Tết cùng Shopee nhận ngay quà hấp dẫn!"}
{"audio": "dataset/ad002.wav", "sentence": "Dầu gội Head & Shoulders đánh bay gàu tức thì."}
{"audio": "dataset/ad003.wav", "sentence": "Khám phá hương vị mới cùng Pepsi vị chanh!"}
```

**Lưu ý:**
- Mỗi dòng là một JSON object
- `audio`: Đường dẫn đến file audio (relative hoặc absolute)
- `sentence`: Text transcription tương ứng

### Tạo JSONL từ dữ liệu có sẵn

#### Ví dụ 1: Từ VIVOS dataset

```bash
python prepare_jsonl.py \
    --audio-dir "archive/vivos/train/waves" \
    --prompts-file "archive/vivos/train/prompts.txt" \
    --output "train.jsonl" \
    --dataset-name "dataset"
```

#### Ví dụ 2: Từ thư mục audio và file text riêng

Nếu bạn có:
- Thư mục `my_audio/` chứa các file `.wav`
- File `transcriptions.txt` với format: `filename text`

Tạo file `prompts.txt`:
```bash
# Format: FILENAME TEXT
ad001 Sắm Tết cùng Shopee nhận ngay quà hấp dẫn!
ad002 Dầu gội Head & Shoulders đánh bay gàu tức thì.
```

Sau đó chạy:
```bash
python prepare_jsonl.py \
    --audio-dir "my_audio" \
    --prompts-file "transcriptions.txt" \
    --output "train.jsonl" \
    --dataset-name "dataset"
```

### Yêu Cầu Về Dữ Liệu

- **Số lượng tối thiểu**: 100 samples (khuyến nghị 1000+)
- **Độ dài audio**: 1-30 giây (tối ưu: 5-15 giây)
- **Format audio**: WAV, MP3, FLAC, M4A, OGG
- **Sample rate**: Tự động resample về 16kHz
- **Chất lượng**: Audio rõ ràng, ít noise
- **Text**: Chính xác, không có lỗi chính tả

---

## ✅ Validate Dữ Liệu

**Luôn validate dữ liệu trước khi fine-tuning!**

```bash
python validate_dataset.py \
    --jsonl-file "train.jsonl" \
    --audio-dir "archive/vivos/train/waves"
```

Script sẽ kiểm tra:
- ✅ Format JSON hợp lệ
- ✅ File audio tồn tại
- ✅ Text không rỗng
- ✅ Audio có thể đọc được
- ✅ Thống kê độ dài audio và text

**Output mẫu:**
```
============================================================
VALIDATION RESULTS
============================================================

Tổng số entries: 1000
Entries hợp lệ: 998
Entries không hợp lệ: 2

Thống kê text:
  - Độ dài trung bình: 45.2 ký tự
  - Độ dài min: 5 ký tự
  - Độ dài max: 120 ký tự

Thống kê audio:
  - Độ dài trung bình: 8.5 giây
  - Tổng thời lượng: 141.67 phút (2.36 giờ)

✅ Dataset hợp lệ! Sẵn sàng cho fine-tuning.
```

---

## 🚀 Fine-tuning

### Các Model Có Sẵn

1. **Whisper (OpenAI)**:
   - `openai/whisper-tiny` - Nhỏ nhất, nhanh nhất
   - `openai/whisper-base` - Cân bằng (khuyến nghị)
   - `openai/whisper-small` - Chính xác hơn
   - `openai/whisper-medium` - Rất chính xác
   - `openai/whisper-large-v3` - Chính xác nhất (lớn nhất)

2. **PhoWhisper (VinAI - tối ưu cho tiếng Việt)**:
   - `vinai/PhoWhisper-base` - Khuyến nghị cho tiếng Việt
   - `vinai/PhoWhisper-large` - Chính xác nhất cho tiếng Việt

### Fine-tuning Cơ Bản

```bash
python fine_tune_whisper.py \
    --model-name "vinai/PhoWhisper-base" \
    --train-jsonl "train.jsonl" \
    --audio-dir "archive/vivos/train/waves" \
    --output-dir "./whisper-finetuned" \
    --num-epochs 3 \
    --batch-size 16 \
    --learning-rate 1e-5
```

### Fine-tuning Nâng Cao

```bash
python fine_tune_whisper.py \
    --model-name "vinai/PhoWhisper-large" \
    --train-jsonl "train.jsonl" \
    --audio-dir "archive/vivos/train/waves" \
    --eval-jsonl "test.jsonl" \
    --output-dir "./whisper-finetuned-large" \
    --num-epochs 5 \
    --batch-size 8 \
    --learning-rate 5e-6 \
    --warmup-steps 1000 \
    --gradient-accumulation-steps 2 \
    --fp16
```

### Tham Số Quan Trọng

| Tham số | Mô tả | Giá trị mặc định | Khuyến nghị |
|---------|-------|------------------|-------------|
| `--model-name` | Model base để fine-tune | `openai/whisper-base` | `vinai/PhoWhisper-base` cho tiếng Việt |
| `--num-epochs` | Số lần train qua toàn bộ dataset | 3 | 3-5 |
| `--batch-size` | Số samples mỗi batch | 16 | 8-32 (tùy GPU) |
| `--learning-rate` | Learning rate | 1e-5 | 1e-5 đến 5e-6 |
| `--fp16` | Mixed precision (nhanh hơn, ít VRAM hơn) | False | Bật nếu có GPU |
| `--gradient-accumulation-steps` | Tích lũy gradient | 1 | 2-4 nếu batch size nhỏ |

### Với GPU

Nếu có GPU, thêm `--fp16` để tăng tốc:

```bash
python fine_tune_whisper.py \
    --model-name "vinai/PhoWhisper-base" \
    --train-jsonl "train.jsonl" \
    --audio-dir "archive/vivos/train/waves" \
    --output-dir "./whisper-finetuned" \
    --num-epochs 3 \
    --batch-size 16 \
    --fp16
```

### Với CPU (chậm hơn nhiều)

Giảm batch size và số epochs:

```bash
python fine_tune_whisper.py \
    --model-name "vinai/PhoWhisper-base" \
    --train-jsonl "train.jsonl" \
    --audio-dir "archive/vivos/train/waves" \
    --output-dir "./whisper-finetuned" \
    --num-epochs 2 \
    --batch-size 4
```

### Thời Gian Training (Ước Tính)

| Dataset Size | GPU | CPU |
|--------------|-----|-----|
| 100 samples | ~5 phút | ~30 phút |
| 1,000 samples | ~30 phút | ~3 giờ |
| 10,000 samples | ~3 giờ | ~30 giờ |

*Với model `PhoWhisper-base`, batch size 16, 3 epochs*

---

## 🎯 Sử Dụng Model Đã Fine-tune

### Trong Python

```python
from transformers import pipeline
import torch

# Load model đã fine-tune
pipe = pipeline(
    "automatic-speech-recognition",
    model="./whisper-finetuned",  # Đường dẫn đến model đã fine-tune
    device=0 if torch.cuda.is_available() else -1,
)

# Transcribe audio
result = pipe("path/to/audio.wav")
print(result["text"])
```

### Trong app.py

Sửa hàm `transcribe_audio()` để ưu tiên model đã fine-tune:

```python
# Thay đổi model path
_phowhisper_pipe = pipeline(
    "automatic-speech-recognition",
    model="./whisper-finetuned",  # Model đã fine-tune
    device=0 if torch.cuda.is_available() else -1,
)
```

---

## 🔧 Troubleshooting

### Lỗi: Out of Memory (OOM)

**Giải pháp:**
1. Giảm `--batch-size` (ví dụ: 16 → 8 → 4)
2. Tăng `--gradient-accumulation-steps` (ví dụ: 1 → 2 → 4)
3. Sử dụng model nhỏ hơn (`base` thay vì `large`)
4. Bật `--fp16` nếu có GPU

### Lỗi: CUDA out of memory

**Giải pháp:**
```bash
# Giảm batch size
--batch-size 4

# Hoặc sử dụng CPU
# (bỏ --fp16 và giảm batch size)
```

### Training quá chậm

**Giải pháp:**
1. Sử dụng GPU nếu có
2. Bật `--fp16`
3. Tăng `--batch-size` (nếu đủ VRAM)
4. Giảm số epochs hoặc dataset size

### Model không cải thiện

**Giải pháp:**
1. Kiểm tra chất lượng dữ liệu training
2. Tăng số epochs
3. Điều chỉnh learning rate (thử 5e-6 hoặc 1e-6)
4. Thêm nhiều dữ liệu training
5. Sử dụng model lớn hơn (`large` thay vì `base`)

### Lỗi: File audio không tìm thấy

**Giải pháp:**
1. Kiểm tra đường dẫn trong JSONL
2. Đảm bảo `--audio-dir` đúng
3. Chạy `validate_dataset.py` để kiểm tra

### Lỗi: Module not found

**Giải pháp:**
```bash
pip install -r requirements.txt
```

---

## 📊 Best Practices

1. **Chia dữ liệu**: 80% train, 20% validation
2. **Augmentation**: Có thể thêm noise, speed variation (tùy chọn)
3. **Early stopping**: Dừng khi validation loss không giảm
4. **Checkpoint**: Lưu checkpoint thường xuyên
5. **Evaluation**: Đánh giá trên test set riêng

---

## 📚 Tài Liệu Tham Khảo

- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [PhoWhisper trên Hugging Face](https://huggingface.co/vinai/PhoWhisper-base)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)

---

## 💡 Tips

- Bắt đầu với model `PhoWhisper-base` - đã được tối ưu cho tiếng Việt
- Fine-tune với ít dữ liệu trước (100-500 samples) để test
- Sử dụng GPU nếu có thể - nhanh hơn 10-20 lần
- Validate dữ liệu kỹ trước khi train
- Lưu checkpoint để có thể tiếp tục training nếu bị gián đoạn

---

**Chúc bạn fine-tuning thành công! 🎉**



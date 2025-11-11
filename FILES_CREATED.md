# 📁 Các File Đã Tạo Cho Fine-tuning

## Scripts Chính

### 1. `prepare_jsonl.py`
**Mục đích**: Chuyển đổi dữ liệu audio và text thành format JSONL chuẩn

**Sử dụng**:
```bash
python prepare_jsonl.py \
    --audio-dir "archive/vivos/train/waves" \
    --prompts-file "archive/vivos/train/prompts.txt" \
    --output "train.jsonl" \
    --dataset-name "dataset"
```

**Chức năng**:
- Đọc file prompts.txt với format: `FILENAME TEXT`
- Tìm các file audio tương ứng
- Tạo file JSONL với format: `{"audio": "path", "sentence": "text"}`

---

### 2. `validate_dataset.py`
**Mục đích**: Kiểm tra và validate dữ liệu trước khi fine-tuning

**Sử dụng**:
```bash
python validate_dataset.py \
    --jsonl-file "train.jsonl" \
    --audio-dir "archive/vivos/train/waves"
```

**Chức năng**:
- Kiểm tra format JSON hợp lệ
- Kiểm tra file audio tồn tại
- Kiểm tra text không rỗng
- Kiểm tra audio có thể đọc được
- Hiển thị thống kê (độ dài text, thời lượng audio)

---

### 3. `fine_tune_whisper.py`
**Mục đích**: Script chính để fine-tune Whisper/PhoWhisper

**Sử dụng**:
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

**Chức năng**:
- Load model Whisper/PhoWhisper từ Hugging Face
- Load và xử lý dataset từ JSONL
- Fine-tune model với các tham số tùy chỉnh
- Lưu model đã fine-tune

**Tham số chính**:
- `--model-name`: Model base (vinai/PhoWhisper-base, openai/whisper-base, ...)
- `--train-jsonl`: File JSONL training data
- `--audio-dir`: Thư mục chứa audio files
- `--output-dir`: Thư mục lưu model
- `--num-epochs`: Số epochs (mặc định: 3)
- `--batch-size`: Batch size (mặc định: 16)
- `--learning-rate`: Learning rate (mặc định: 1e-5)
- `--fp16`: Mixed precision training (cho GPU)
- `--eval-jsonl`: File JSONL evaluation (optional)

---

## Tài Liệu

### 4. `FINE_TUNING_GUIDE.md`
**Hướng dẫn chi tiết đầy đủ** về:
- Yêu cầu hệ thống
- Cài đặt
- Chuẩn bị dữ liệu
- Validate dữ liệu
- Fine-tuning (cơ bản và nâng cao)
- Sử dụng model đã fine-tune
- Troubleshooting
- Best practices

### 5. `QUICK_START_FINETUNE.md`
**Hướng dẫn nhanh** để bắt đầu trong 5 phút:
- Các bước cơ bản
- Lệnh mẫu
- Format JSONL
- Troubleshooting nhanh

---

## Scripts Ví Dụ

### 6. `example_prepare_vivos.sh`
Script bash ví dụ để chuẩn bị dữ liệu VIVOS:
- Tạo train.jsonl và test.jsonl
- Validate datasets

### 7. `example_finetune.sh`
Script bash ví dụ để fine-tune:
- Fine-tune với PhoWhisper-base
- Các tham số mặc định

---

## File Cấu Hình

### 8. `requirements.txt` (đã cập nhật)
Đã thêm các thư viện cần thiết:
- `datasets>=2.14.0` - Để load và xử lý dataset
- `accelerate>=0.20.0` - Để tăng tốc training
- `librosa>=0.10.0` - Xử lý audio
- `soundfile>=0.12.0` - Đọc file audio

---

## Workflow Đề Xuất

1. **Chuẩn bị dữ liệu**:
   ```bash
   python prepare_jsonl.py --audio-dir ... --prompts-file ... --output train.jsonl
   ```

2. **Validate**:
   ```bash
   python validate_dataset.py --jsonl-file train.jsonl --audio-dir ...
   ```

3. **Fine-tune**:
   ```bash
   python fine_tune_whisper.py --model-name vinai/PhoWhisper-base --train-jsonl train.jsonl ...
   ```

4. **Sử dụng model**:
   ```python
   from transformers import pipeline
   pipe = pipeline("automatic-speech-recognition", model="./whisper-finetuned")
   ```

---

## Format JSONL Chuẩn

Mỗi dòng trong file JSONL:

```json
{"audio": "dataset/ad001.wav", "sentence": "Sắm Tết cùng Shopee nhận ngay quà hấp dẫn!"}
{"audio": "dataset/ad002.wav", "sentence": "Dầu gội Head & Shoulders đánh bay gàu tức thì."}
{"audio": "dataset/ad003.wav", "sentence": "Khám phá hương vị mới cùng Pepsi vị chanh!"}
```

**Lưu ý**:
- `audio`: Đường dẫn đến file audio (relative hoặc absolute)
- `sentence`: Text transcription tương ứng
- Mỗi dòng là một JSON object hợp lệ

---

## Models Có Sẵn

### PhoWhisper (Khuyến nghị cho tiếng Việt):
- `vinai/PhoWhisper-base` - Cân bằng, nhanh
- `vinai/PhoWhisper-large` - Chính xác nhất

### Whisper (OpenAI):
- `openai/whisper-tiny` - Nhỏ nhất
- `openai/whisper-base` - Cân bằng
- `openai/whisper-small` - Chính xác hơn
- `openai/whisper-medium` - Rất chính xác
- `openai/whisper-large-v3` - Chính xác nhất

---

## Lưu Ý

- Luôn validate dữ liệu trước khi fine-tune
- Bắt đầu với model nhỏ (base) để test
- Sử dụng GPU nếu có thể (nhanh hơn 10-20 lần)
- Fine-tune với ít dữ liệu trước (100-500 samples) để test workflow
- Xem `FINE_TUNING_GUIDE.md` để biết chi tiết đầy đủ

---

**Chúc bạn fine-tuning thành công! 🎉**



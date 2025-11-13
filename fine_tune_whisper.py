"""
Script fine-tuning PhoWhisper với dữ liệu JSONL

Sử dụng thư viện transformers và datasets từ Hugging Face
Chỉ hỗ trợ PhoWhisper models (vinai/PhoWhisper-base, vinai/PhoWhisper-large)
"""

import os
import json
import argparse
from pathlib import Path
import torch
import numpy as np
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import TrainerCallback


def load_jsonl_dataset(jsonl_file, audio_dir, use_negative_samples=False):
    """
    Load dataset từ JSONL file
    Hỗ trợ dataset có cả positive (is_match=True) và negative (is_match=False) samples
    
    Args:
        jsonl_file: File JSONL chứa {"audio": "path", "sentence": "text", "is_match": bool}
        audio_dir: Thư mục chứa audio files (base directory, ví dụ: archive/vivos/train/waves)
        use_negative_samples: Nếu True, sẽ train cả negative samples. Mặc định False (chỉ train positive)
    
    Returns:
        Dataset với audio paths đã được resolve thành absolute paths
    """
    # Validate inputs
    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"JSONL file không tồn tại: {jsonl_file}")
    
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory không tồn tại: {audio_dir}")
    
    print(f"Đang load dataset từ: {jsonl_file}")
    print(f"Audio directory: {audio_dir}")
    
    # Đọc JSONL file
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Không thể parse dòng JSON: {line[:50]}... Error: {e}")
    
    print(f"Đã đọc {len(data)} entries từ JSONL")
    
    # Convert audio_dir to absolute path
    audio_dir = os.path.abspath(audio_dir)
    
    # Phân loại và kiểm tra entries
    positive_data = []
    negative_data = []
    missing_files = []
    
    for idx, item in enumerate(data):
        if 'audio' not in item or 'sentence' not in item:
            print(f"Warning: Entry {idx} thiếu 'audio' hoặc 'sentence': {item}")
            continue
        
        # Kiểm tra is_match (mặc định True nếu không có field này - tương thích ngược)
        is_match = item.get('is_match', True)
        
        audio_path = item['audio']
        # Normalize path: thay backslash bằng forward slash (cross-platform)
        audio_path = audio_path.replace('\\', '/')
        
        # Nếu là relative path, thêm audio_dir vào
        if not os.path.isabs(audio_path):
            # Join với audio_dir và normalize path
            audio_path = os.path.join(audio_dir, audio_path)
            audio_path = os.path.normpath(audio_path)
        else:
            audio_path = os.path.normpath(audio_path)
        
        # Kiểm tra file có tồn tại không
        if os.path.exists(audio_path):
            # Lưu absolute path
            item['audio'] = os.path.abspath(audio_path)
            if is_match:
                positive_data.append(item)
            else:
                negative_data.append(item)
        else:
            missing_files.append(audio_path)
            if len(missing_files) <= 10:  # Chỉ hiển thị 10 file đầu tiên
                print(f"Warning: File không tồn tại: {audio_path}")
    
    if missing_files:
        print(f"Warning: Tổng cộng {len(missing_files)} files không tồn tại (đã hiển thị 10 đầu tiên)")
    
    # Thống kê
    print(f"\nPhân loại entries:")
    print(f"  - Positive (is_match=True): {len(positive_data)}")
    print(f"  - Negative (is_match=False): {len(negative_data)}")
    
    # Chọn data để train
    if use_negative_samples:
        valid_data = positive_data + negative_data
        print(f"  - Sẽ train trên cả positive và negative: {len(valid_data)} samples")
    else:
        valid_data = positive_data
        print(f"  - Chỉ train trên positive samples: {len(valid_data)} samples")
        if len(negative_data) > 0:
            print(f"  - Bỏ qua {len(negative_data)} negative samples (dùng --use-negative-samples để train cả negative)")
    
    if len(valid_data) == 0:
        raise ValueError(f"Không có entries hợp lệ nào để train! Kiểm tra lại paths trong JSONL và audio_dir.")
    
    # Tạo dataset từ JSONL file với absolute paths
    # KHÔNG load audio vào memory - chỉ lưu paths để xử lý on-the-fly
    from datasets import Dataset
    from pathlib import Path
    
    # Tạo dataset trực tiếp từ list of dicts
    # Giữ audio paths dưới dạng strings (không load audio vào memory)
    print("Đang tạo dataset từ data (chỉ lưu paths, không load audio)...")
    
    # Chuẩn bị data với audio paths dưới dạng strings
    dataset_data = []
    for item in valid_data:
        # Đảm bảo audio path là absolute path string
        audio_path = str(Path(item["audio"]).resolve())
        dataset_data.append({
            "audio": audio_path,  # Lưu path string, không load audio
            "sentence": item["sentence"]
        })
    
    # Tạo Dataset object từ list of dicts
    # Audio column là strings (paths), không phải Audio objects
    dataset = Dataset.from_list(dataset_data)
    
    print("Dataset đã sẵn sàng (audio sẽ được load on-the-fly trong training)")
    
    return dataset


def prepare_dataset(batch, processor):
    """Chuẩn bị dữ liệu cho training"""
    # Load và resample audio (batch processing)
    audio_arrays = []
    sampling_rates = []
    sentences = []
    
    # Xử lý từng item trong batch
    for item in batch["audio"]:
        audio_arrays.append(item["array"])
        sampling_rates.append(item["sampling_rate"])
    
    for sentence in batch["sentence"]:
        sentences.append(sentence)
    
    # Compute log-Mel input features từ audio arrays
    inputs = processor.feature_extractor(
        audio_arrays, 
        sampling_rate=sampling_rates[0] if sampling_rates else 16000,
        return_tensors="np"
    ).input_features
    
    # Encode target text thành label ids
    labels = processor.tokenizer(
        sentences,
        return_tensors="np",
        padding=True,
        truncation=True
    ).input_ids
    
    # Replace padding token id's của the labels bằng -100 để ignore trong loss
    labels = [
        [(label if label != processor.tokenizer.pad_token_id else -100) for label in label_ids]
        for label_ids in labels
    ]
    
    return {
        "input_features": inputs.tolist(),
        "labels": labels
    }


def compute_wer(reference, hypothesis):
    """Tính Word Error Rate (WER) sử dụng dynamic programming"""
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()
    
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    
    # Dynamic programming để tính edit distance
    n, m = len(ref_words), len(hyp_words)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    # Initialize
    for i in range(n + 1):
        dp[i][0] = i  # deletions
    for j in range(m + 1):
        dp[0][j] = j  # insertions
    
    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + 1     # substitution
                )
    
    return dp[n][m] / n


def compute_cer(reference, hypothesis):
    """Tính Character Error Rate (CER)"""
    ref_chars = list(reference.strip().lower().replace(' ', ''))
    hyp_chars = list(hypothesis.strip().lower().replace(' ', ''))
    
    if len(ref_chars) == 0:
        return 1.0 if len(hyp_chars) > 0 else 0.0
    
    n, m = len(ref_chars), len(hyp_chars)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 1)
    
    return dp[n][m] / n


def compute_metrics(pred, processor):
    """Tính toán các metrics: WER, CER, và các chỉ số khác"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Decode predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Normalize text
    normalizer = BasicTextNormalizer()
    pred_str = [normalizer(pred) for pred in pred_str]
    label_str = [normalizer(label) for label in label_str]
    
    # Tính các metrics
    wer_scores = []
    cer_scores = []
    exact_matches = 0
    total_samples = len(pred_str)
    
    for pred, label in zip(pred_str, label_str):
        wer = compute_wer(label, pred)
        cer = compute_cer(label, pred)
        wer_scores.append(wer)
        cer_scores.append(cer)
        
        # Exact match (case-insensitive)
        if pred.strip().lower() == label.strip().lower():
            exact_matches += 1
    
    # Tính trung bình
    avg_wer = np.mean(wer_scores) if wer_scores else 1.0
    avg_cer = np.mean(cer_scores) if cer_scores else 1.0
    accuracy = exact_matches / total_samples if total_samples > 0 else 0.0
    
    # Word-level accuracy
    total_words = 0
    correct_words = 0
    for pred, label in zip(pred_str, label_str):
        pred_words = pred.strip().lower().split()
        label_words = label.strip().lower().split()
        total_words += len(label_words)
        min_len = min(len(pred_words), len(label_words))
        correct_words += sum(1 for i in range(min_len) if pred_words[i] == label_words[i])
    
    word_accuracy = correct_words / total_words if total_words > 0 else 0.0
    
    return {
        "wer": avg_wer,
        "cer": avg_cer,
        "accuracy": accuracy,
        "word_accuracy": word_accuracy,
        "exact_matches": exact_matches,
        "total_samples": total_samples
    }


def format_metrics_table(metrics_dict, dataset_name="Evaluation"):
    """Tạo bảng metrics đẹp để hiển thị"""
    lines = []
    lines.append("=" * 70)
    lines.append(f"📊 BẢNG KẾT QUẢ ĐÁNH GIÁ: {dataset_name}")
    lines.append("=" * 70)
    
    # Định nghĩa các metrics và mô tả
    metric_info = [
        ("WER (Word Error Rate)", "wer", "Tỷ lệ lỗi từ (càng thấp càng tốt, 0.0 = hoàn hảo)"),
        ("CER (Character Error Rate)", "cer", "Tỷ lệ lỗi ký tự (càng thấp càng tốt, 0.0 = hoàn hảo)"),
        ("Accuracy (Exact Match)", "accuracy", "Tỷ lệ câu chính xác hoàn toàn (càng cao càng tốt, 1.0 = hoàn hảo)"),
        ("Word Accuracy", "word_accuracy", "Tỷ lệ từ chính xác (càng cao càng tốt, 1.0 = hoàn hảo)"),
    ]
    
    # Hiển thị từng metric
    for metric_name, metric_key, description in metric_info:
        if metric_key in metrics_dict:
            value = metrics_dict[metric_key]
            if isinstance(value, float):
                if metric_key in ["wer", "cer"]:
                    # WER và CER: hiển thị dưới dạng phần trăm và số thập phân
                    lines.append(f"\n{metric_name}:")
                    lines.append(f"  Giá trị: {value:.4f} ({value*100:.2f}%)")
                    lines.append(f"  Mô tả: {description}")
                    # Đánh giá chất lượng
                    if value < 0.1:
                        quality = "Xuất sắc ⭐⭐⭐⭐⭐"
                    elif value < 0.2:
                        quality = "Rất tốt ⭐⭐⭐⭐"
                    elif value < 0.3:
                        quality = "Tốt ⭐⭐⭐"
                    elif value < 0.5:
                        quality = "Khá ⭐⭐"
                    else:
                        quality = "Cần cải thiện ⭐"
                    lines.append(f"  Đánh giá: {quality}")
                else:
                    # Accuracy: hiển thị dưới dạng phần trăm
                    lines.append(f"\n{metric_name}:")
                    lines.append(f"  Giá trị: {value:.4f} ({value*100:.2f}%)")
                    lines.append(f"  Mô tả: {description}")
                    # Đánh giá chất lượng
                    if value > 0.9:
                        quality = "Xuất sắc ⭐⭐⭐⭐⭐"
                    elif value > 0.8:
                        quality = "Rất tốt ⭐⭐⭐⭐"
                    elif value > 0.7:
                        quality = "Tốt ⭐⭐⭐"
                    elif value > 0.5:
                        quality = "Khá ⭐⭐"
                    else:
                        quality = "Cần cải thiện ⭐"
                    lines.append(f"  Đánh giá: {quality}")
    
    # Thông tin bổ sung
    if "exact_matches" in metrics_dict and "total_samples" in metrics_dict:
        exact = metrics_dict["exact_matches"]
        total = metrics_dict["total_samples"]
        lines.append(f"\n📈 Thống kê:")
        lines.append(f"  Số mẫu đánh giá: {total}")
        lines.append(f"  Số câu chính xác hoàn toàn: {exact}")
        lines.append(f"  Số câu có lỗi: {total - exact}")
    
    lines.append("=" * 70)
    return "\n".join(lines)


def save_detailed_report(results, output_dir, model_name):
    """Lưu báo cáo chi tiết ra file text và JSON"""
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Tạo báo cáo text
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("📋 BÁO CÁO KẾT QUẢ HUẤN LUYỆN MÔ HÌNH")
    report_lines.append("=" * 70)
    report_lines.append(f"Model: {model_name}")
    report_lines.append(f"Thời gian: {timestamp}")
    report_lines.append(f"Thư mục output: {output_dir}")
    report_lines.append("")
    
    # Thêm từng phần đánh giá
    if "eval_during_training" in results:
        report_lines.append(format_metrics_table(
            results["eval_during_training"], 
            "Validation Set (Trong quá trình training)"
        ))
        report_lines.append("")
    
    if "eval_after_training" in results:
        eval_info = results["eval_after_training"]
        dataset_path = eval_info.get("path", "Unknown")
        metrics = eval_info.get("metrics", {})
        report_lines.append(format_metrics_table(
            metrics,
            f"Test Set (Sau khi training) - {os.path.basename(dataset_path)}"
        ))
        report_lines.append("")
    
    # Tổng kết
    report_lines.append("=" * 70)
    report_lines.append("📌 TỔNG KẾT")
    report_lines.append("=" * 70)
    
    best_wer = float('inf')
    best_dataset = None
    
    if "eval_during_training" in results:
        wer = results["eval_during_training"].get("wer", float('inf'))
        if wer < best_wer:
            best_wer = wer
            best_dataset = "Validation Set"
    
    if "eval_after_training" in results:
        wer = results["eval_after_training"].get("metrics", {}).get("wer", float('inf'))
        if wer < best_wer:
            best_wer = wer
            best_dataset = "Test Set"
    
    if best_dataset:
        report_lines.append(f"WER tốt nhất: {best_wer:.4f} ({best_wer*100:.2f}%) trên {best_dataset}")
        if best_wer < 0.1:
            report_lines.append("🎉 Mô hình đạt chất lượng xuất sắc!")
        elif best_wer < 0.2:
            report_lines.append("✅ Mô hình đạt chất lượng rất tốt!")
        elif best_wer < 0.3:
            report_lines.append("👍 Mô hình đạt chất lượng tốt!")
        else:
            report_lines.append("⚠️  Mô hình cần được cải thiện thêm.")
    
    report_lines.append("=" * 70)
    
    # Lưu file text
    text_report_path = os.path.join(output_dir, "evaluation_report.txt")
    try:
        with open(text_report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"✅ Đã lưu báo cáo text: {text_report_path}")
    except Exception as e:
        print(f"⚠️  Không thể lưu báo cáo text: {e}")
    
    # Lưu file JSON (đã có sẵn trong code chính)
    json_report_path = os.path.join(output_dir, "training_report.json")
    try:
        # Thêm metadata vào JSON
        json_results = {
            "model_name": model_name,
            "timestamp": timestamp,
            "output_dir": output_dir,
            "results": results
        }
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã lưu báo cáo JSON: {json_report_path}")
    except Exception as e:
        print(f"⚠️  Không thể lưu báo cáo JSON: {e}")
    
    # In báo cáo ra console
    print("\n" + "\n".join(report_lines))


class WhisperDataCollator:
    """Custom data collator xử lý audio on-the-fly và tối ưu VRAM"""
    
    def __init__(self, processor, tokenizer, padding=True, device="cuda", enable_cache=False):
        self.processor = processor
        self.tokenizer = tokenizer
        self.padding = padding
        self.device = device if torch.cuda.is_available() else "cpu"
        # Tắt cache để tiết kiệm VRAM (cache audio tốn nhiều RAM)
        self.enable_cache = enable_cache
        self._audio_cache = {} if enable_cache else {}  # Cache chỉ dùng khi enable_cache=True
        self._cache_size_limit = 0  # Tắt cache để tiết kiệm VRAM
    
    def _load_audio(self, audio_path, max_duration=30.0):
        """
        Load audio từ file với error handling - tối ưu RAM
        Chỉ load tối đa max_duration giây (mặc định 30s = 3000 frames mel spectrogram)
        """
        # Tắt cache để tiết kiệm VRAM
        if self.enable_cache and audio_path in self._audio_cache:
            return self._audio_cache[audio_path]
        
        try:
            # Sử dụng librosa để load audio
            import librosa
            # Load với resampling trực tiếp và giới hạn độ dài để tiết kiệm memory
            # max_duration=30s vì Whisper chỉ cần 3000 frames (30s * 100 frames/s)
            audio, sr = librosa.load(
                audio_path, 
                sr=16000, 
                mono=True,
                dtype=np.float32,  # Sử dụng float32 thay vì float64 để tiết kiệm memory
                duration=max_duration  # Chỉ load tối đa 30 giây để tiết kiệm RAM
            )
            
            # Đảm bảo audio không quá dài (an toàn thêm)
            max_samples = int(max_duration * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            # Không cache để tiết kiệm VRAM
            return audio, sr
        except Exception as e:
            print(f"Warning: Không thể load audio {audio_path}: {e}")
            # Tạo zero array nếu không load được (1 giây audio)
            return np.zeros(16000, dtype=np.float32), 16000
    
    def __call__(self, features):
        """
        Xử lý batch: load audio từ paths và convert thành features
        Tối ưu để sử dụng VRAM thay vì RAM
        """
        # Tách audio paths và sentences từ features
        if isinstance(features[0], dict):
            audio_paths = [f["audio"] for f in features]
            sentences = [f["sentence"] for f in features]
        else:
            # Fallback nếu format khác
            audio_paths = [str(f.get("audio", "")) for f in features]
            sentences = [str(f.get("sentence", "")) for f in features]
        
        # Tối ưu RAM: Process và xóa ngay từng audio thay vì giữ tất cả trong memory
        # Compute log-Mel input features trên CPU - xử lý streaming để tiết kiệm RAM
        processed_features = []
        
        for idx, audio_path in enumerate(audio_paths):
            try:
                # Load audio
                audio_array, sr = self._load_audio(audio_path)
                
                # Extract features ngay lập tức
                features = self.processor.feature_extractor(
                    audio_array,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features  # Shape: [1, n_mels, time_frames]
                
                # Xóa audio array ngay sau khi extract features để giải phóng RAM
                del audio_array
                
                # Đảm bảo có đúng 3000 frames (Whisper yêu cầu)
                current_length = features.shape[-1]
                if current_length < 3000:
                    # Pad với zeros ở cuối
                    padding = torch.zeros(
                        1, 
                        features.shape[1], 
                        3000 - current_length,
                        dtype=features.dtype
                    )
                    features = torch.cat([features, padding], dim=-1)
                elif current_length > 3000:
                    # Truncate đến 3000
                    features = features[:, :, :3000]
                
                # Đảm bảo shape cuối cùng là [1, n_mels, 3000]
                assert features.shape[-1] == 3000, f"Feature length must be 3000, got {features.shape[-1]}"
                
                # Lưu feature đã processed (chỉ giữ feature, không giữ audio)
                processed_features.append(features.squeeze(0))  # Remove batch dim: [n_mels, 3000]
                
            except Exception as e:
                print(f"Warning: Lỗi khi xử lý audio {idx}: {e}")
                # Tạo zero features với shape đúng [n_mels, 3000]
                n_mels = 80  # Whisper sử dụng 80 mel bins
                zero_features = torch.zeros(n_mels, 3000, dtype=torch.float32)
                processed_features.append(zero_features)
        
        # Stack tất cả features lại thành batch: [batch_size, n_mels, 3000]
        input_features = torch.stack(processed_features, dim=0)
        
        # Xóa processed_features list để giải phóng RAM
        del processed_features
        
        # Final check: đảm bảo shape đúng
        assert input_features.shape[-1] == 3000, f"All features must have length 3000, got {input_features.shape}"
        
        # Encode labels với max_length nhỏ hơn để tiết kiệm VRAM
        labels = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256  # Giảm từ 448 xuống 256 để tiết kiệm VRAM
        ).input_ids
        
        # Replace padding tokens với -100 để ignore trong loss
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        labels = labels.masked_fill(labels == pad_token_id, -100)
        
        # Batch - Trainer sẽ tự động move lên GPU khi cần
        # Không move ở đây để Trainer có thể quản lý device tốt hơn
        batch = {
            "input_features": input_features,
            "labels": labels
        }
        
        # Force garbage collection để giải phóng RAM ngay lập tức
        import gc
        gc.collect()
        
        return batch


class ClearCacheCallback(TrainerCallback):
    """Callback để clear GPU cache định kỳ (mỗi N steps để không làm chậm)"""
    def __init__(self, clear_interval=50):
        self.clear_interval = clear_interval
        self.step_count = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        # Chỉ clear cache mỗi N steps để không làm chậm training
        if self.step_count % self.clear_interval == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return control


def main():
    parser = argparse.ArgumentParser(description='Fine-tune PhoWhisper model')
    parser.add_argument('--model-name', type=str, default='vinai/PhoWhisper-base',
                        help='PhoWhisper model name (default: vinai/PhoWhisper-base). Có thể dùng: vinai/PhoWhisper-base, vinai/PhoWhisper-large')
    parser.add_argument('--train-jsonl', type=str, default='data/train.jsonl',
                        help='File JSONL training data (default: data/train.jsonl)')
    parser.add_argument('--audio-dir', type=str, default='archive/vivos/train/waves',
                        help='Thư mục chứa audio files (default: archive/vivos/train/waves)')
    parser.add_argument('--output-dir', type=str, default='./phowhisper-finetuned',
                        help='Thư mục lưu model sau khi fine-tune (default: ./phowhisper-finetuned)')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='Số epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4, phù hợp GPU 8GB)')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5)')
    parser.add_argument('--warmup-steps', type=int, default=200,
                        help='Warmup steps (default: 200)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                        help='Gradient accumulation steps (default: 8, giữ effective batch nhỏ trên GPU 8GB)')
    parser.add_argument('--per-device-eval-batch-size', type=int, default=None,
                        help='Batch size cho evaluation (default: giống batch size train)')
    parser.add_argument('--fp16', action='store_true', default=None,
                        help='Sử dụng mixed precision training (FP16) - MẶC ĐỊNH BẬT khi có GPU')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false',
                        help='Tắt FP16 (mixed precision training)')
    parser.add_argument('--max-speed', action='store_true', default=None,
                        help='Tối ưu tối đa tốc độ training (tăng batch size, dataloader). Mặc định tắt để tiết kiệm VRAM.')
    parser.add_argument('--no-max-speed', dest='max_speed', action='store_false', default=None,
                        help='Tắt chế độ tối ưu tốc độ (dùng cấu hình tiết kiệm VRAM)')
    parser.add_argument('--auto-batch-size', action='store_true', default=None,
                        help='Tự động tìm batch size tối ưu dựa trên VRAM (mặc định tắt để tránh OOM trên GPU nhỏ)')
    parser.add_argument('--no-auto-batch-size', dest='auto_batch_size', action='store_false', default=None,
                        help='Tắt tự động tìm batch size tối ưu')
    parser.add_argument('--eval-jsonl', type=str, default='data/dev.jsonl',
                        help='File JSONL evaluation data (default: data/dev.jsonl, set None để tắt evaluation)')
    parser.add_argument('--eval-after-train-jsonl', type=str, default='',
                        help='Đánh giá thêm sau khi train trên JSONL khác (ví dụ: data/test.jsonl). Bỏ trống để bỏ qua')
    parser.add_argument('--num-beams', type=int, default=2,
                        help='Beam size khi generate (mặc định 2)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing factor (mặc định 0.1)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Số mẫu mỗi lần train theo từng đợt (mặc định: 1000, 0 = tắt chunked training)')
    parser.add_argument('--start-chunk', type=int, default=0,
                        help='Bắt đầu từ chunk index nào (mặc định 0)')
    parser.add_argument('--no-eval', action='store_true', default=True,
                        help='Tắt evaluation trong và sau khi training (MẶC ĐỊNH BẬT)')
    parser.add_argument('--eval', dest='no_eval', action='store_false',
                        help='Bật evaluation (tắt --no-eval)')
    parser.add_argument('--use-negative-samples', action='store_true',
                        help='Train cả negative samples (is_match=False). Mặc định chỉ train positive samples')
    
    args = parser.parse_args()
    
    # Xác định thông tin GPU để điều chỉnh cấu hình phù hợp
    gpu_total_mem_gb = None
    if torch.cuda.is_available():
        try:
            gpu_total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            gpu_total_mem_gb = None
    
    # Nếu user không chỉ định, đặt mặc định thân thiện với GPU 8GB
    if args.fp16 is None:
        args.fp16 = torch.cuda.is_available()
    
    if args.max_speed is None:
        # Mặc định tắt max-speed để tránh tăng batch size trên GPU nhỏ
        args.max_speed = False
    
    if args.auto_batch_size is None:
        # Mặc định tắt auto batch size để tránh thử batch quá lớn
        args.auto_batch_size = False
    
    # Validate model name - chỉ cho phép PhoWhisper models
    valid_phowhisper_models = ['vinai/PhoWhisper-base', 'vinai/PhoWhisper-large']
    if args.model_name not in valid_phowhisper_models:
        print(f"Warning: Model '{args.model_name}' không phải là PhoWhisper model.")
        print(f"Chỉ hỗ trợ: {', '.join(valid_phowhisper_models)}")
        print(f"Sử dụng default: vinai/PhoWhisper-base")
        args.model_name = 'vinai/PhoWhisper-base'
    
    # Validate file paths
    if not os.path.exists(args.train_jsonl):
        raise FileNotFoundError(
            f"Train JSONL file không tồn tại: {args.train_jsonl}\n"
            f"Hãy chạy split_dataset.py trước để tạo dataset, hoặc chỉ định đúng path với --train-jsonl"
        )
    
    # Convert to absolute paths
    args.train_jsonl = os.path.abspath(args.train_jsonl)
    args.audio_dir = os.path.abspath(args.audio_dir)
    
    if args.eval_jsonl and args.eval_jsonl.lower() != 'none':
        if not os.path.exists(args.eval_jsonl):
            print(f"Warning: Eval JSONL file không tồn tại: {args.eval_jsonl}")
            print(f"Sẽ tiếp tục training không có evaluation")
            args.eval_jsonl = None
        else:
            args.eval_jsonl = os.path.abspath(args.eval_jsonl)
    else:
        args.eval_jsonl = None
    
    # Xử lý tối ưu tốc độ
    if args.max_speed:
        print("\n🚀 Chế độ MAX SPEED được bật - Tối ưu tối đa tốc độ training")
        # Tự động bật FP16
        if torch.cuda.is_available() and not args.fp16:
            args.fp16 = True
            print("  ✓ FP16 (Mixed Precision) được bật tự động")
        elif torch.cuda.is_available() and args.fp16:
            print("  ✓ FP16 (Mixed Precision) đã được bật")
        
        # Tăng batch size nếu có thể (nhưng vẫn an toàn)
        original_bs = args.batch_size
        if args.batch_size <= 8:
            args.batch_size = 16
            print(f"  ✓ Batch size được tăng lên: {original_bs} → {args.batch_size}")
        elif args.batch_size >= 16:
            print(f"  ✓ Batch size đã được tối ưu: {args.batch_size}")
        
        # Giảm gradient accumulation nếu batch size đã tăng
        if args.batch_size >= 16 and args.gradient_accumulation_steps > 2:
            args.gradient_accumulation_steps = 2
            print(f"  ✓ Gradient accumulation được điều chỉnh: {args.gradient_accumulation_steps}")
        
        # Tăng learning rate một chút để học nhanh hơn
        if args.learning_rate <= 1e-5:
            args.learning_rate = 1.5e-5
            print(f"  ✓ Learning rate được tăng lên: {args.learning_rate}")
    
    # Tự động tìm batch size tối ưu
    if args.auto_batch_size and torch.cuda.is_available():
        print("\n🔍 Tự động tìm batch size tối ưu...")
        original_batch_size = args.batch_size
        optimal_batch_size = original_batch_size
        
        # Xác định batch size tối đa dựa trên dung lượng GPU
        if gpu_total_mem_gb is not None:
            if gpu_total_mem_gb <= 8.5:
                max_auto_batch = max(4, original_batch_size)
            elif gpu_total_mem_gb <= 12.5:
                max_auto_batch = max(8, original_batch_size)
            else:
                max_auto_batch = max(16, original_batch_size)
        else:
            max_auto_batch = max(8, original_batch_size)
        
        # Thử tăng batch size dần cho đến khi chạm ngưỡng an toàn
        test_batch_sizes = []
        current_bs = original_batch_size
        while current_bs <= max_auto_batch:
            if current_bs not in test_batch_sizes:
                test_batch_sizes.append(current_bs)
            next_bs = current_bs * 2
            if next_bs <= max_auto_batch and next_bs != current_bs:
                current_bs = next_bs
            else:
                break
        
        if not test_batch_sizes:
            test_batch_sizes = [original_batch_size]
        
        for test_bs in test_batch_sizes:
            try:
                # Test với một batch nhỏ
                torch.cuda.empty_cache()
                test_tensor = torch.randn(test_bs, 80, 3000, device='cuda', dtype=torch.float16 if args.fp16 else torch.float32)
                del test_tensor
                torch.cuda.empty_cache()
                optimal_batch_size = test_bs
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    break
                else:
                    raise
        
        if optimal_batch_size > original_batch_size:
            args.batch_size = optimal_batch_size
            print(f"  ✓ Batch size tối ưu được tìm thấy: {optimal_batch_size} (tăng từ {original_batch_size})")
        else:
            print(f"  ✓ Batch size hiện tại là tối ưu: {original_batch_size}")
    
    # Nếu không chỉ định eval batch size, dùng cùng batch size train
    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = args.batch_size
    
    # Kiểm tra GPU và dependencies
    print("\n=== Kiểm tra môi trường ===")
    
    # Kiểm tra GPU
    print(f"PyTorch Version: {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"✓ GPU được phát hiện!")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Số GPU: {torch.cuda.device_count()}")
    else:
        print("⚠ Warning: Không có GPU được phát hiện!")
        print("  Training sẽ chạy trên CPU (rất chậm)")
        print("\n  Kiểm tra:")
        print("    1. GPU có được kết nối không?")
        print("    2. Driver GPU đã được cài đặt chưa? (chạy: nvidia-smi)")
        print("    3. PyTorch có hỗ trợ CUDA không?")
        
        # Kiểm tra nvidia-smi
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("\n  ⚠ nvidia-smi chạy được nhưng PyTorch không detect GPU")
                print("  → PyTorch không được build với CUDA support")
                print("\n  Giải pháp: Cài lại PyTorch với CUDA")
                print("  pip uninstall torch torchvision torchaudio")
                print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            else:
                print("\n  ⚠ Không thể chạy nvidia-smi")
                print("  → Có thể GPU driver chưa được cài đặt")
        except FileNotFoundError:
            print("\n  ⚠ Không tìm thấy nvidia-smi")
            print("  → Có thể GPU driver chưa được cài đặt hoặc không có NVIDIA GPU")
        except Exception as e:
            print(f"\n  ⚠ Lỗi khi kiểm tra nvidia-smi: {e}")
        
        print("\n  Nếu không có GPU, training vẫn có thể chạy trên CPU nhưng rất chậm")
        # Không hỏi user input trong non-interactive mode, chỉ cảnh báo
        print("  ⚠ Tiếp tục với CPU (có thể rất chậm)...")
        print("  💡 Khuyến nghị: Cài PyTorch với CUDA để sử dụng GPU")
        print("     pip uninstall torch torchvision torchaudio")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    # Kiểm tra dependencies
    try:
        import librosa
        print(f"✓ librosa: {librosa.__version__}")
    except ImportError:
        raise ImportError(
            "Thiếu thư viện librosa. Cài đặt bằng lệnh:\n"
            "pip install librosa soundfile"
        )
    
    try:
        import soundfile
        print(f"✓ soundfile: {soundfile.__version__}")
    except ImportError:
        raise ImportError(
            "Thiếu thư viện soundfile. Cài đặt bằng lệnh:\n"
            "pip install soundfile"
        )
    
    print(f"========================\n")
    
    print(f"\n=== Cấu hình Training ===")
    print(f"Model: {args.model_name}")
    print(f"Train JSONL: {args.train_jsonl}")
    print(f"Audio Directory: {args.audio_dir}")
    if args.eval_jsonl:
        print(f"Eval JSONL: {args.eval_jsonl}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Eval Batch Size: {args.per_device_eval_batch_size}")
    print(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"FP16 (Mixed Precision): {args.fp16 and torch.cuda.is_available()}")
    if args.max_speed:
        print(f"🚀 MAX SPEED Mode: BẬT")
    if args.auto_batch_size:
        print(f"🔍 Auto Batch Size: BẬT")
    print(f"GPU: {device}")
    print(f"========================\n")
    
    # Load processor và model
    print(f"Đang load PhoWhisper model: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(args.model_name, language="vi", task="transcribe")
    
    # Load model - Trainer sẽ tự động move model lên GPU nếu có
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    # Set language và task tokens
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
    
    if torch.cuda.is_available():
        print(f"Model sẽ được tự động chuyển lên GPU khi training bắt đầu")
    else:
        print("Model sẽ chạy trên CPU")
    
    # Load dataset - KHÔNG preprocess để tránh out of memory
    # Sẽ xử lý audio on-the-fly trong data collator
    print("Đang load dataset (không preprocess để tiết kiệm RAM)...")
    train_dataset = load_jsonl_dataset(args.train_jsonl, args.audio_dir, use_negative_samples=args.use_negative_samples)
    print(f"Train dataset: {len(train_dataset)} samples")
    
    # Load eval dataset nếu có và không tắt evaluation
    eval_dataset = None
    if args.eval_jsonl and not args.no_eval:
        eval_dataset = load_jsonl_dataset(args.eval_jsonl, args.audio_dir, use_negative_samples=False)
        print(f"Eval dataset: {len(eval_dataset)} samples")
    elif args.no_eval:
        print("Evaluation đã được tắt (--no-eval)")
        args.eval_jsonl = None  # Đảm bảo không load eval dataset
    
    print("Dataset sẽ được xử lý on-the-fly trong training để tiết kiệm RAM")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        gradient_checkpointing=True,  # Tiết kiệm VRAM
        fp16=args.fp16 and torch.cuda.is_available(),  # Chỉ dùng FP16 nếu có GPU
        bf16=False,  # Có thể dùng bf16 nếu GPU hỗ trợ (A100, H100)
        # Dataloader settings: tối ưu cho tốc độ nếu max-speed, ngược lại tối ưu VRAM
        dataloader_num_workers=2 if (args.max_speed and torch.cuda.is_available()) else 0,
        dataloader_pin_memory=True if (args.max_speed and torch.cuda.is_available()) else False,
        dataloader_prefetch_factor=2 if (args.max_speed and torch.cuda.is_available()) else None,
        eval_strategy="no" if args.no_eval else ("steps" if eval_dataset else "no"),
        eval_steps=None if args.no_eval else (500 if eval_dataset else None),
        save_steps=500,
        logging_steps=100,
        report_to="none",
        load_best_model_at_end=True if eval_dataset else False,
        push_to_hub=False,
        # GPU settings
        remove_unused_columns=False,
        # Optimizations
        optim="adamw_torch",  # Sử dụng AdamW optimizer
        max_grad_norm=1.0,  # Gradient clipping
        # Generation for evaluation
        predict_with_generate=True if eval_dataset else False,
        generation_num_beams=max(1, args.num_beams),
    )
    
    # Log GPU memory nếu có
    if torch.cuda.is_available():
        print(f"\n=== GPU Memory Info ===")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_memory = total_memory - (torch.cuda.memory_reserved(0) / 1e9)
        print(f"GPU Memory Total: {total_memory:.2f} GB")
        print(f"GPU Memory Free: {free_memory:.2f} GB")
        print(f"========================\n")
    else:
        print("\n⚠ Training sẽ chạy trên CPU - rất chậm!")
        print("  Khuyến nghị: Sử dụng GPU để training nhanh hơn\n")
    
    # Custom Data Collator để xử lý audio on-the-fly và tối ưu VRAM
    # Sử dụng GPU để xử lý features và giảm RAM usage
    # Tạo data collator với device - tắt cache để tiết kiệm VRAM
    device_for_collator = "cuda" if torch.cuda.is_available() else "cpu"
    data_collator = WhisperDataCollator(
        processor=processor,
        tokenizer=processor.tokenizer,
        padding=True,
        device=device_for_collator,
        enable_cache=False  # Tắt cache để tiết kiệm VRAM
    )
    
    # Trainer với tối ưu memory
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=None,  # sẽ gán sau (hỗ trợ train theo từng chunk)
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor) if eval_dataset else None,
    )
    
    # Clear cache trước khi training để giải phóng memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Đã clear GPU cache")
    
    import gc
    gc.collect()
    print("✓ Đã clear RAM cache")
    print()
    
    # Train với callback để clear cache định kỳ
    # ClearCacheCallback đã được định nghĩa ở module level để tránh lỗi pickle
    # Train (hỗ trợ train theo từng đợt/chunk)
    print("Bắt đầu training...")
    print("Lưu ý: Audio được load on-the-fly để tiết kiệm RAM, sử dụng VRAM của GPU")
    print("\n💡 Tối ưu RAM đã được áp dụng:")
    print("   - Load và xóa audio ngay sau khi extract features (streaming processing)")
    print("   - Chỉ load tối đa 30 giây audio mỗi file (đủ cho Whisper)")
    print("   - Sử dụng float32 thay vì float64")
    print("   - Không cache audio trong RAM")
    print("   - Force garbage collection sau mỗi batch")
    print("   → Giảm RAM usage đáng kể so với cách load toàn bộ audio trước")
    
    if args.max_speed:
        print("🚀 Chế độ MAX SPEED: Tối ưu tối đa tốc độ training")
        print(f"   - Batch size: {args.batch_size}")
        print(f"   - FP16: {'BẬT' if args.fp16 else 'TẮT'}")
        print(f"   - Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
        print(f"   - Dataloader workers: 2 (tăng tốc)")
    else:
        print("⚠ Tối ưu VRAM: batch_size=8, gradient_accumulation=4, cache tắt, workers=0")
        print("⚠ Nếu muốn tăng tốc độ, thử:")
        print("   1. Dùng --max-speed để tự động tối ưu")
        print("   2. Dùng --auto-batch-size để tự động tìm batch size tối ưu")
        print("   3. Bật --fp16 để tăng tốc")
        print("⚠ Nếu vẫn hết VRAM, thử:")
        print("   1. Giảm batch_size xuống 4 hoặc 2: --batch-size 4")
        print("   2. Tăng gradient_accumulation_steps: --gradient-accumulation-steps 8")
        print("   3. Set environment variable: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    # Thêm callback để clear cache mỗi 50 steps
    trainer.add_callback(ClearCacheCallback(clear_interval=50))

    # Xử lý train theo chunk nếu được yêu cầu
    if args.chunk_size and args.chunk_size > 0:
        total_samples = len(train_dataset)
        num_chunks = (total_samples + args.chunk_size - 1) // args.chunk_size
        start_chunk = max(0, min(args.start_chunk, num_chunks - 1))

        print(f"\n=== Train theo từng đợt (Chunked Training) ===")
        print(f"Tổng samples: {total_samples}")
        print(f"Chunk size: {args.chunk_size} samples/chunk")
        print(f"Tổng số chunk: {num_chunks}")
        print(f"Bắt đầu từ chunk: {start_chunk}")
        if args.no_eval:
            print(f"⚠️  Evaluation đã tắt - Train liên tục qua các chunks, không test")
        print(f"→ Sẽ train liên tục từ chunk {start_chunk} đến chunk {num_chunks - 1}")

        from pathlib import Path as _Path
        checkpoints_root = _Path(args.output_dir) / "chunk_checkpoints"
        checkpoints_root.mkdir(parents=True, exist_ok=True)

        # Train lần lượt từng chunk
        for chunk_idx in range(start_chunk, num_chunks):
            start_idx = chunk_idx * args.chunk_size
            end_idx = min(start_idx + args.chunk_size, total_samples)
            indices = list(range(start_idx, end_idx))
            chunk_dataset = train_dataset.select(indices)

            print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks}: samples [{start_idx}:{end_idx}) ---")
            trainer.train_dataset = chunk_dataset

            # Không resume Trainer state giữa các chunk (tránh trạng thái "đã hoàn tất")
            # Model weights đã được cập nhật sau mỗi chunk nên tiếp tục train trực tiếp
            trainer.train(resume_from_checkpoint=False)

            # Lưu checkpoint theo chunk để an toàn
            chunk_ckpt_dir = checkpoints_root / f"chunk_{chunk_idx:04d}"
            chunk_ckpt_dir.mkdir(parents=True, exist_ok=True)
            print(f"Lưu checkpoint chunk vào: {str(chunk_ckpt_dir)}")
            trainer.save_model(str(chunk_ckpt_dir))
            trainer.save_state()
            # Lưu processor (tokenizer/feature extractor) một lần ở output_dir chính
            try:
                processor.save_pretrained(args.output_dir)
            except Exception:
                pass

            # Clear cache giữa các chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc as _gc
            _gc.collect()

        print("\nĐã hoàn tất train theo từng đợt.")
    else:
        # Train thông thường trên toàn bộ dataset (chỉ khi chunk_size = 0)
        trainer.train_dataset = train_dataset
        trainer.train()
    
    # Save model
    print(f"Đang lưu model vào: {args.output_dir}")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
    # Đánh giá sau khi train và lưu báo cáo (chỉ nếu không tắt evaluation)
    if not args.no_eval:
        print("\n" + "="*70)
        print("📊 BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH")
        print("="*70 + "\n")
        
        results = {}
        if eval_dataset:
            try:
                print("Đang đánh giá trên validation set...")
                eval_metrics = trainer.evaluate()
                results["eval_during_training"] = eval_metrics
                print("✅ Hoàn thành đánh giá validation set")
            except Exception as e:
                print(f"⚠️  Warning: Không thể evaluate trên eval_dataset: {e}")

        # Đánh giá bổ sung trên JSONL khác (ví dụ test)
        if args.eval_after_train_jsonl:
            try:
                extra_path = os.path.abspath(args.eval_after_train_jsonl)
                if os.path.exists(extra_path):
                    print(f"\nĐang đánh giá bổ sung trên: {os.path.basename(extra_path)}")
                    extra_dataset = load_jsonl_dataset(extra_path, args.audio_dir, use_negative_samples=False)
                    # Tạo một Trainer tạm để predict trên extra dataset
                    extra_trainer = Seq2SeqTrainer(
                        args=training_args,
                        model=trainer.model,
                        eval_dataset=extra_dataset,
                        tokenizer=processor.feature_extractor,
                        data_collator=data_collator,
                        compute_metrics=lambda pred: compute_metrics(pred, processor),
                    )
                    extra_metrics = extra_trainer.evaluate(eval_dataset=extra_dataset)
                    results["eval_after_training"] = {
                        "path": extra_path,
                        "metrics": extra_metrics,
                    }
                    print("✅ Hoàn thành đánh giá test set")
                else:
                    print(f"⚠️  Warning: eval_after_train_jsonl không tồn tại: {extra_path}")
            except Exception as e:
                print(f"⚠️  Warning: Không thể evaluate bổ sung: {e}")

        # Tạo và lưu báo cáo chi tiết
        if results:
            print("\n" + "="*70)
            print("📋 TẠO BÁO CÁO KẾT QUẢ")
            print("="*70 + "\n")
            save_detailed_report(results, args.output_dir, args.model_name)
        else:
            print("⚠️  Không có kết quả đánh giá để tạo báo cáo")
    else:
        print("\n⚠️  Evaluation đã được tắt (--no-eval) - Bỏ qua đánh giá")

    print("\n" + "="*70)
    print("✅ HOÀN THÀNH FINE-TUNING!")
    print("="*70)
    print(f"📁 Model đã được lưu tại: {args.output_dir}")
    if not args.no_eval:
        print(f"📊 Báo cáo đã được lưu tại:")
        print(f"   - {os.path.join(args.output_dir, 'evaluation_report.txt')}")
        print(f"   - {os.path.join(args.output_dir, 'training_report.json')}")
    print("="*70)


if __name__ == '__main__':
    main()


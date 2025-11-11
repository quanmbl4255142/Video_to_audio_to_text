"""
Script fine-tuning Whisper/PhoWhisper với dữ liệu JSONL

Sử dụng thư viện transformers và datasets từ Hugging Face
"""

import os
import json
import argparse
from pathlib import Path
import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def load_jsonl_dataset(jsonl_file, audio_dir):
    """
    Load dataset từ JSONL file
    
    Args:
        jsonl_file: File JSONL chứa {"audio": "path", "sentence": "text"}
        audio_dir: Thư mục chứa audio files (base directory)
    """
    print(f"Đang load dataset từ: {jsonl_file}")
    
    # Đọc JSONL file
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"Đã đọc {len(data)} entries")
    
    # Kiểm tra và sửa đường dẫn audio
    valid_data = []
    for item in data:
        audio_path = item['audio']
        # Nếu là relative path, thêm audio_dir vào
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(audio_dir, audio_path)
        
        # Kiểm tra file có tồn tại không
        if os.path.exists(audio_path):
            item['audio'] = audio_path
            valid_data.append(item)
        else:
            print(f"Warning: File không tồn tại: {audio_path}")
    
    print(f"Số entries hợp lệ: {len(valid_data)}")
    
    # Tạo dataset từ JSONL file
    # Tạo temporary JSONL với absolute paths
    import tempfile
    temp_jsonl = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
    
    for item in data:
        audio_path = item['audio']
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(audio_dir, audio_path)
        if os.path.exists(audio_path):
            item['audio'] = audio_path
            temp_jsonl.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    temp_jsonl.close()
    
    # Load dataset từ temporary file
    dataset = load_dataset('json', data_files={'train': temp_jsonl.name}, split='train')
    
    # Load audio
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Xóa temporary file
    os.unlink(temp_jsonl.name)
    
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


def compute_metrics(pred, processor):
    """Tính toán metrics (WER - Word Error Rate)"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Decode predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Normalize text
    normalizer = BasicTextNormalizer()
    pred_str = [normalizer(pred) for pred in pred_str]
    label_str = [normalizer(label) for label in label_str]
    
    # Tính WER (đơn giản - có thể cải thiện)
    wer = 0.0
    for pred, label in zip(pred_str, label_str):
        pred_words = pred.split()
        label_words = label.split()
        if len(label_words) > 0:
            # Tính số từ sai
            errors = sum(1 for p, l in zip(pred_words, label_words) if p != l)
            wer += errors / len(label_words)
        else:
            wer += 1.0 if pred_words else 0.0
    
    wer = wer / len(pred_str) if pred_str else 1.0
    
    return {"wer": wer}


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Whisper/PhoWhisper model')
    parser.add_argument('--model-name', type=str, default='openai/whisper-base',
                        help='Model name (default: openai/whisper-base). Có thể dùng: vinai/PhoWhisper-base, vinai/PhoWhisper-large')
    parser.add_argument('--train-jsonl', type=str, required=True,
                        help='File JSONL training data')
    parser.add_argument('--audio-dir', type=str, required=True,
                        help='Thư mục chứa audio files')
    parser.add_argument('--output-dir', type=str, default='./whisper-finetuned',
                        help='Thư mục lưu model sau khi fine-tune (default: ./whisper-finetuned)')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='Số epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5)')
    parser.add_argument('--warmup-steps', type=int, default=500,
                        help='Warmup steps (default: 500)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--fp16', action='store_true',
                        help='Sử dụng mixed precision training (FP16)')
    parser.add_argument('--eval-jsonl', type=str, default=None,
                        help='File JSONL evaluation data (optional)')
    
    args = parser.parse_args()
    
    # Kiểm tra GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng device: {device}")
    if device == "cpu":
        print("Warning: Không có GPU, training sẽ chậm hơn nhiều")
    
    # Load processor và model
    print(f"Đang load model: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(args.model_name, language="vi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    # Set language và task tokens
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
    
    # Load dataset
    train_dataset = load_jsonl_dataset(args.train_jsonl, args.audio_dir)
    
    # Prepare dataset
    print("Đang chuẩn bị dataset...")
    def prepare_fn(examples):
        return prepare_dataset(examples, processor)
    
    train_dataset = train_dataset.map(
        prepare_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Preparing dataset"
    )
    
    # Load eval dataset nếu có
    eval_dataset = None
    if args.eval_jsonl:
        eval_dataset = load_jsonl_dataset(args.eval_jsonl, args.audio_dir)
        eval_dataset = eval_dataset.map(
            prepare_fn,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Preparing eval dataset"
        )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        gradient_checkpointing=True,
        fp16=args.fp16,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=500 if eval_dataset else None,
        save_steps=500,
        logging_steps=100,
        report_to="none",
        load_best_model_at_end=True if eval_dataset else False,
        push_to_hub=False,
    )
    
    # Data collator
    from transformers import DataCollatorForSeq2Seq
    
    data_collator = DataCollatorForSeq2Seq(
        processor=processor,
        model=model,
        padding=True
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor) if eval_dataset else None,
    )
    
    # Train
    print("Bắt đầu training...")
    trainer.train()
    
    # Save model
    print(f"Đang lưu model vào: {args.output_dir}")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
    print("Hoàn thành fine-tuning!")


if __name__ == '__main__':
    main()


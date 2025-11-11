"""
Script để chuyển đổi dữ liệu audio và text thành format JSONL chuẩn cho fine-tuning Whisper/PhoWhisper

Format JSONL:
{"audio": "dataset/ad001.wav", "sentence": "Sắm Tết cùng Shopee nhận ngay quà hấp dẫn!"}
{"audio": "dataset/ad002.wav", "sentence": "Dầu gội Head & Shoulders đánh bay gàu tức thì."}
"""

import os
import json
import argparse
from pathlib import Path


def load_prompts(prompts_file):
    """Đọc file prompts.txt và trả về dictionary {filename: text}"""
    prompts = {}
    if not os.path.exists(prompts_file):
        print(f"Warning: File {prompts_file} không tồn tại")
        return prompts
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Format: FILENAME TEXT
            parts = line.split(' ', 1)
            if len(parts) == 2:
                filename = parts[0]
                text = parts[1]
                prompts[filename] = text
            else:
                print(f"Warning: Dòng không đúng format: {line}")
    
    return prompts


def find_audio_files(audio_dir):
    """Tìm tất cả file audio trong thư mục (recursive)"""
    audio_files = {}
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                filepath = os.path.join(root, file)
                # Lấy tên file không có extension làm key
                filename_without_ext = os.path.splitext(file)[0]
                audio_files[filename_without_ext] = filepath
    
    return audio_files


def create_jsonl_dataset(audio_dir, prompts_file, output_file, dataset_name="dataset"):
    """
    Tạo file JSONL từ audio files và prompts
    
    Args:
        audio_dir: Thư mục chứa audio files (có thể có subdirectories)
        prompts_file: File prompts.txt chứa text transcriptions
        output_file: File JSONL output
        dataset_name: Tên thư mục dataset trong JSONL (để relative path)
    """
    print(f"Đang đọc prompts từ: {prompts_file}")
    prompts = load_prompts(prompts_file)
    print(f"Đã đọc {len(prompts)} prompts")
    
    print(f"Đang tìm audio files trong: {audio_dir}")
    audio_files = find_audio_files(audio_dir)
    print(f"Đã tìm thấy {len(audio_files)} audio files")
    
    # Tạo mapping: filename -> (audio_path, text)
    matched = []
    unmatched_audio = []
    unmatched_prompts = []
    
    for filename, text in prompts.items():
        if filename in audio_files:
            audio_path = audio_files[filename]
            # Tạo relative path từ dataset_name
            relative_path = os.path.join(dataset_name, os.path.relpath(audio_path, audio_dir))
            matched.append({
                "audio": relative_path,
                "sentence": text
            })
        else:
            unmatched_prompts.append(filename)
    
    for filename in audio_files:
        if filename not in prompts:
            unmatched_audio.append(filename)
    
    print(f"\nKết quả:")
    print(f"  - Số cặp audio-text khớp: {len(matched)}")
    print(f"  - Audio không có prompt: {len(unmatched_audio)}")
    print(f"  - Prompt không có audio: {len(unmatched_prompts)}")
    
    if unmatched_prompts and len(unmatched_prompts) <= 10:
        print(f"\n  Ví dụ prompts không có audio: {unmatched_prompts[:5]}")
    if unmatched_audio and len(unmatched_audio) <= 10:
        print(f"\n  Ví dụ audio không có prompt: {unmatched_audio[:5]}")
    
    # Ghi file JSONL
    print(f"\nĐang ghi file JSONL: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in matched:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Đã tạo file JSONL với {len(matched)} entries")
    return len(matched)


def main():
    parser = argparse.ArgumentParser(description='Chuyển đổi dữ liệu audio-text sang JSONL format')
    parser.add_argument('--audio-dir', type=str, required=True,
                        help='Thư mục chứa audio files')
    parser.add_argument('--prompts-file', type=str, required=True,
                        help='File prompts.txt chứa transcriptions')
    parser.add_argument('--output', type=str, default='train.jsonl',
                        help='File JSONL output (default: train.jsonl)')
    parser.add_argument('--dataset-name', type=str, default='dataset',
                        help='Tên thư mục dataset trong JSONL (default: dataset)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_dir):
        print(f"Error: Thư mục audio không tồn tại: {args.audio_dir}")
        return
    
    create_jsonl_dataset(
        args.audio_dir,
        args.prompts_file,
        args.output,
        args.dataset_name
    )


if __name__ == '__main__':
    main()



"""
Script để validate và kiểm tra dữ liệu training trước khi fine-tuning
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter


def validate_jsonl(jsonl_file, audio_dir):
    """
    Validate JSONL dataset
    
    Kiểm tra:
    - Format JSON hợp lệ
    - File audio tồn tại
    - Text không rỗng
    - Audio có thể đọc được
    """
    print(f"Đang validate dataset: {jsonl_file}")
    print(f"Audio directory: {audio_dir}\n")
    
    errors = []
    warnings = []
    stats = {
        'total': 0,
        'valid': 0,
        'missing_audio': 0,
        'empty_text': 0,
        'invalid_json': 0,
        'audio_errors': 0,
    }
    
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    text_lengths = []
    audio_durations = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            stats['total'] += 1
            
            # Parse JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                stats['invalid_json'] += 1
                continue
            
            # Kiểm tra keys
            if 'audio' not in data:
                errors.append(f"Line {line_num}: Missing 'audio' key")
                continue
            
            if 'sentence' not in data:
                errors.append(f"Line {line_num}: Missing 'sentence' key")
                continue
            
            # Kiểm tra text
            text = data['sentence'].strip()
            if not text:
                warnings.append(f"Line {line_num}: Empty text")
                stats['empty_text'] += 1
            else:
                text_lengths.append(len(text))
            
            # Kiểm tra audio file
            audio_path = data['audio']
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(audio_dir, audio_path)
            
            if not os.path.exists(audio_path):
                errors.append(f"Line {line_num}: Audio file not found: {audio_path}")
                stats['missing_audio'] += 1
                continue
            
            # Kiểm tra extension
            ext = os.path.splitext(audio_path)[1].lower()
            if ext not in audio_extensions:
                warnings.append(f"Line {line_num}: Unusual audio extension: {ext}")
            
            # Kiểm tra file size
            try:
                file_size = os.path.getsize(audio_path)
                if file_size == 0:
                    errors.append(f"Line {line_num}: Audio file is empty: {audio_path}")
                    stats['audio_errors'] += 1
                    continue
            except Exception as e:
                errors.append(f"Line {line_num}: Cannot read audio file: {e}")
                stats['audio_errors'] += 1
                continue
            
            # Thử đọc audio (basic check)
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(audio_path)
                duration = len(audio) / 1000.0  # seconds
                audio_durations.append(duration)
                
                if duration < 0.1:
                    warnings.append(f"Line {line_num}: Audio too short ({duration:.2f}s): {audio_path}")
                if duration > 60:
                    warnings.append(f"Line {line_num}: Audio very long ({duration:.2f}s): {audio_path}")
            except Exception as e:
                warnings.append(f"Line {line_num}: Cannot load audio (may still be valid): {e}")
            
            stats['valid'] += 1
    
    # Print results
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"\nTổng số entries: {stats['total']}")
    print(f"Entries hợp lệ: {stats['valid']}")
    print(f"Entries không hợp lệ: {stats['total'] - stats['valid']}")
    print(f"\nChi tiết lỗi:")
    print(f"  - Missing audio files: {stats['missing_audio']}")
    print(f"  - Empty text: {stats['empty_text']}")
    print(f"  - Invalid JSON: {stats['invalid_json']}")
    print(f"  - Audio errors: {stats['audio_errors']}")
    
    if text_lengths:
        print(f"\nThống kê text:")
        print(f"  - Độ dài trung bình: {sum(text_lengths) / len(text_lengths):.1f} ký tự")
        print(f"  - Độ dài min: {min(text_lengths)} ký tự")
        print(f"  - Độ dài max: {max(text_lengths)} ký tự")
    
    if audio_durations:
        print(f"\nThống kê audio:")
        print(f"  - Độ dài trung bình: {sum(audio_durations) / len(audio_durations):.2f} giây")
        print(f"  - Độ dài min: {min(audio_durations):.2f} giây")
        print(f"  - Độ dài max: {max(audio_durations):.2f} giây")
        total_duration = sum(audio_durations)
        print(f"  - Tổng thời lượng: {total_duration / 60:.2f} phút ({total_duration / 3600:.2f} giờ)")
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for error in errors[:20]:  # Show first 20 errors
            print(f"  {error}")
        if len(errors) > 20:
            print(f"  ... và {len(errors) - 20} lỗi khác")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings[:20]:  # Show first 20 warnings
            print(f"  {warning}")
        if len(warnings) > 20:
            print(f"  ... và {len(warnings) - 20} cảnh báo khác")
    
    print("\n" + "=" * 60)
    if stats['valid'] == stats['total'] and not errors:
        print("✅ Dataset hợp lệ! Sẵn sàng cho fine-tuning.")
        return True
    else:
        print("❌ Dataset có lỗi. Vui lòng sửa trước khi fine-tuning.")
        return False


def main():
    parser = argparse.ArgumentParser(description='Validate JSONL dataset cho fine-tuning')
    parser.add_argument('--jsonl-file', type=str, required=True,
                        help='File JSONL cần validate')
    parser.add_argument('--audio-dir', type=str, required=True,
                        help='Thư mục chứa audio files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.jsonl_file):
        print(f"Error: File không tồn tại: {args.jsonl_file}")
        return
    
    if not os.path.exists(args.audio_dir):
        print(f"Error: Thư mục không tồn tại: {args.audio_dir}")
        return
    
    validate_jsonl(args.jsonl_file, args.audio_dir)


if __name__ == '__main__':
    main()



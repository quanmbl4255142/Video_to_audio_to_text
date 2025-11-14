"""
Script để chia dataset thành 3 tập: train (80%), test (10%), dev (10%)

Input: archive/vivos/train/ (prompts.txt + waves/)
Output: train.jsonl, test.jsonl, dev.jsonl
"""

import os
import json
import argparse
import random
from pathlib import Path


def load_prompts(prompts_file):
    """
    Đọc file transcript và trả về dictionary {filename: text}
    Hỗ trợ 2 format:
    1. Format mới (pipe-separated): FILENAME|TEXT|TIMESTAMPS
    2. Format cũ (space-separated): FILENAME TEXT
    """
    prompts = {}
    if not os.path.exists(prompts_file):
        print(f"Error: File {prompts_file} không tồn tại")
        return prompts
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Thử format mới trước (pipe-separated): FILENAME|TEXT|TIMESTAMPS
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    filename = parts[0].strip()
                    text = parts[1].strip()
                    # Bỏ qua timestamps (parts[2]) nếu có
                    # Lưu filename không có extension để match với audio files
                    filename_without_ext = os.path.splitext(filename)[0]
                    prompts[filename_without_ext] = text
                else:
                    print(f"Warning: Dòng {line_num} không đúng format (pipe): {line}")
            else:
                # Format cũ (space-separated): FILENAME TEXT
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    filename = parts[0]
                    text = parts[1]
                    # Lưu filename không có extension để match với audio files
                    filename_without_ext = os.path.splitext(filename)[0]
                    prompts[filename_without_ext] = text
                else:
                    print(f"Warning: Dòng {line_num} không đúng format: {line}")
    
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


def create_dataset_entries(audio_dir, prompts_file, negative_ratio=0.3, seed=42):
    """
    Tạo danh sách các entries (audio, sentence) từ prompts và audio files
    Bao gồm cả positive samples (khớp) và negative samples (không khớp)
    
    Args:
        audio_dir: Thư mục chứa audio files (sẽ dùng làm base path)
        prompts_file: File prompts.txt
        negative_ratio: Tỷ lệ negative samples so với positive (default: 0.3 = 30%)
        seed: Random seed để tạo negative samples reproducible
    
    Returns:
        List of dict: [{"audio": "path", "sentence": "text", "is_match": bool}, ...]
        Path trong JSONL sẽ là relative từ audio_dir
    """
    print(f"Đang đọc prompts từ: {prompts_file}")
    prompts = load_prompts(prompts_file)
    print(f"Đã đọc {len(prompts)} prompts")
    
    print(f"Đang tìm audio files trong: {audio_dir}")
    audio_files = find_audio_files(audio_dir)
    print(f"Đã tìm thấy {len(audio_files)} audio files")
    
    # Tạo positive samples (audio-text khớp)
    positive_entries = []
    unmatched_audio = []
    unmatched_prompts = []
    
    for filename, text in prompts.items():
        if filename in audio_files:
            audio_path = audio_files[filename]
            # Tạo relative path từ audio_dir
            relative_path = os.path.relpath(audio_path, audio_dir)
            relative_path = relative_path.replace('\\', '/')
            
            positive_entries.append({
                "audio": relative_path,
                "sentence": text,
                "is_match": True
            })
        else:
            unmatched_prompts.append((filename, text))
    
    for filename in audio_files:
        if filename not in prompts:
            unmatched_audio.append(filename)
    
    print(f"\nKết quả matching:")
    print(f"  - Số cặp audio-text khớp (positive): {len(positive_entries)}")
    print(f"  - Audio không có prompt: {len(unmatched_audio)}")
    print(f"  - Prompt không có audio: {len(unmatched_prompts)}")
    
    # Tạo negative samples (audio-text không khớp)
    print(f"\nĐang tạo negative samples (tỷ lệ {negative_ratio*100:.1f}% so với positive)...")
    random.seed(seed)
    
    negative_count = int(len(positive_entries) * negative_ratio)
    negative_entries = []
    
    # Tạo negative samples bằng cách ghép audio với text không khớp
    all_audio_keys = list(audio_files.keys())
    all_prompts = list(prompts.items())
    
    created_negative = 0
    max_attempts = negative_count * 10  # Giới hạn số lần thử
    attempts = 0
    
    while created_negative < negative_count and attempts < max_attempts:
        attempts += 1
        
        # Chọn ngẫu nhiên một audio file
        audio_key = random.choice(all_audio_keys)
        audio_path = audio_files[audio_key]
        relative_path = os.path.relpath(audio_path, audio_dir)
        relative_path = relative_path.replace('\\', '/')
        
        # Chọn ngẫu nhiên một text không khớp với audio này
        random_prompt_key, random_text = random.choice(all_prompts)
        
        # Đảm bảo audio và text không khớp
        if audio_key != random_prompt_key:
            negative_entries.append({
                "audio": relative_path,
                "sentence": random_text,
                "is_match": False
            })
            created_negative += 1
    
    print(f"  - Đã tạo {len(negative_entries)} negative samples")
    
    # Kết hợp positive và negative
    all_entries = positive_entries + negative_entries
    
    # Shuffle để trộn positive và negative
    random.shuffle(all_entries)
    
    print(f"\nTổng số entries: {len(all_entries)}")
    print(f"  - Positive (khớp): {len(positive_entries)} ({len(positive_entries)/len(all_entries)*100:.1f}%)")
    print(f"  - Negative (không khớp): {len(negative_entries)} ({len(negative_entries)/len(all_entries)*100:.1f}%)")
    
    return all_entries


def sample_dataset(entries, ratio=1.0, seed=42):
    """
    Lấy một phần của dataset gốc
    
    Args:
        entries: List of dict entries
        ratio: Tỷ lệ dataset để lấy (0.0 - 1.0, default: 1.0 = 100%)
        seed: Random seed để reproducible
    
    Returns:
        Sampled entries
    """
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"Dataset ratio phải trong khoảng (0.0, 1.0], nhưng được {ratio}")
    
    if ratio >= 1.0:
        return entries
    
    random.seed(seed)
    shuffled = entries.copy()
    random.shuffle(shuffled)
    
    sample_size = int(len(shuffled) * ratio)
    sampled = shuffled[:sample_size]
    
    print(f"\nLấy mẫu dataset:")
    print(f"  - Dataset gốc: {len(entries)} entries")
    print(f"  - Tỷ lệ: {ratio*100:.1f}%")
    print(f"  - Dataset sau khi lấy mẫu: {len(sampled)} entries")
    
    return sampled


def split_dataset(entries, train_ratio=0.8, test_ratio=0.1, dev_ratio=0.1, seed=42):
    """
    Chia dataset thành 3 tập: train, test, dev
    Đảm bảo mỗi tập đều có cả positive và negative samples
    
    Args:
        entries: List of dict entries (có field "is_match")
        train_ratio: Tỷ lệ train (default: 0.8)
        test_ratio: Tỷ lệ test (default: 0.1)
        dev_ratio: Tỷ lệ dev (default: 0.1)
        seed: Random seed để reproducible
    
    Returns:
        train_entries, test_entries, dev_entries
    """
    # Validate ratios
    total_ratio = train_ratio + test_ratio + dev_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Tổng tỷ lệ phải bằng 1.0, nhưng được {total_ratio}")
    
    # Tách positive và negative entries
    positive_entries = [e for e in entries if e.get("is_match", True)]
    negative_entries = [e for e in entries if not e.get("is_match", True)]
    
    print(f"\nPhân loại entries:")
    print(f"  - Positive: {len(positive_entries)}")
    print(f"  - Negative: {len(negative_entries)}")
    
    # Shuffle riêng positive và negative với seed
    random.seed(seed)
    shuffled_positive = positive_entries.copy()
    shuffled_negative = negative_entries.copy()
    random.shuffle(shuffled_positive)
    random.shuffle(shuffled_negative)
    
    # Chia positive samples
    pos_total = len(shuffled_positive)
    pos_train_size = int(pos_total * train_ratio)
    pos_test_size = int(pos_total * test_ratio)
    
    pos_train = shuffled_positive[:pos_train_size]
    pos_test = shuffled_positive[pos_train_size:pos_train_size + pos_test_size]
    pos_dev = shuffled_positive[pos_train_size + pos_test_size:]
    
    # Chia negative samples
    neg_total = len(shuffled_negative)
    neg_train_size = int(neg_total * train_ratio)
    neg_test_size = int(neg_total * test_ratio)
    
    neg_train = shuffled_negative[:neg_train_size]
    neg_test = shuffled_negative[neg_train_size:neg_train_size + neg_test_size]
    neg_dev = shuffled_negative[neg_train_size + neg_test_size:]
    
    # Kết hợp positive và negative cho mỗi tập
    train_entries = pos_train + neg_train
    test_entries = pos_test + neg_test
    dev_entries = pos_dev + neg_dev
    
    # Shuffle lại mỗi tập để trộn positive và negative
    random.shuffle(train_entries)
    random.shuffle(test_entries)
    random.shuffle(dev_entries)
    
    total = len(entries)
    
    print(f"\nChia dataset (seed={seed}):")
    print(f"  - Train: {len(train_entries)} ({len(train_entries)/total*100:.1f}%)")
    print(f"    + Positive: {len(pos_train)} ({len(pos_train)/len(train_entries)*100:.1f}%)")
    print(f"    + Negative: {len(neg_train)} ({len(neg_train)/len(train_entries)*100:.1f}%)")
    print(f"  - Test: {len(test_entries)} ({len(test_entries)/total*100:.1f}%)")
    print(f"    + Positive: {len(pos_test)} ({len(pos_test)/len(test_entries)*100:.1f}%)")
    print(f"    + Negative: {len(neg_test)} ({len(neg_test)/len(test_entries)*100:.1f}%)")
    print(f"  - Dev: {len(dev_entries)} ({len(dev_entries)/total*100:.1f}%)")
    print(f"    + Positive: {len(pos_dev)} ({len(pos_dev)/len(dev_entries)*100:.1f}%)")
    print(f"    + Negative: {len(neg_dev)} ({len(neg_dev)/len(dev_entries)*100:.1f}%)")
    print(f"  - Total: {total}")
    
    return train_entries, test_entries, dev_entries


def write_jsonl(entries, output_file):
    """Ghi entries ra file JSONL"""
    print(f"Đang ghi file JSONL: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in entries:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Đã ghi {len(entries)} entries vào {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Chia dataset thành train/test/dev (80/10/10) với cả positive và negative samples')
    parser.add_argument('ratio', type=float, nargs='?', default=None,
                        help='Tỷ lệ dataset gốc để sử dụng (positional argument, ví dụ: 0.1 = 10%%)')
    parser.add_argument('--archive-dir', type=str, default='archive/vivos/train',
                        help='Thư mục chứa dataset (default: archive/vivos/train)')
    parser.add_argument('--transcript-file', type=str, default=None,
                        help='File transcript (default: tự động tìm prompts.txt hoặc transcriptAll.txt trong archive-dir)')
    parser.add_argument('--audio-dir', type=str, default=None,
                        help='Thư mục chứa audio files (default: tự động tìm mp3/ hoặc waves/ trong archive-dir)')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Thư mục output để lưu JSONL files (default: data)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Tỷ lệ train (default: 0.8)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Tỷ lệ test (default: 0.1)')
    parser.add_argument('--dev-ratio', type=float, default=0.1,
                        help='Tỷ lệ dev (default: 0.1)')
    parser.add_argument('--negative-ratio', type=float, default=0.3,
                        help='Tỷ lệ negative samples so với positive (default: 0.3 = 30%%)')
    parser.add_argument('--dataset-ratio', type=float, default=1.0,
                        help='Tỷ lệ dataset gốc để sử dụng (default: 1.0 = 100%%, ví dụ: 0.1 = 10%%) - có thể dùng positional argument thay thế')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed để reproducible (default: 42)')
    args = parser.parse_args()
    
    # Nếu có positional argument, ưu tiên sử dụng nó
    if args.ratio is not None:
        dataset_ratio = args.ratio
    else:
        # Nếu không có positional argument, dùng giá trị từ --dataset-ratio (mặc định 1.0)
        dataset_ratio = args.dataset_ratio
    
    # Validate dataset_ratio
    if dataset_ratio <= 0.0 or dataset_ratio > 1.0:
        raise ValueError(f"Dataset ratio phải trong khoảng (0.0, 1.0], nhưng được {dataset_ratio}")
    
    # Tìm transcript file và waves/ trong archive_dir
    archive_dir = os.path.abspath(args.archive_dir)  # Convert to absolute path
    
    # Xác định file transcript
    transcript_search_paths = []
    archive_root = os.path.dirname(archive_dir.rstrip(os.sep))
    archive_super_root = os.path.dirname(archive_root.rstrip(os.sep)) if archive_root else None

    if args.transcript_file:
        prompts_file = os.path.abspath(args.transcript_file)
        transcript_search_paths.append(prompts_file)
    else:
        transcript_candidates = [
            'transcriptAll.txt',
            'transcript_all.txt',
            'transcript.txt',
            'prompts.txt',
            'prompt.txt',
            'metadata.txt',
            'metadata.csv',
        ]
        found_transcript = None
        search_dirs = []
        for search_dir in [archive_dir, archive_root, archive_super_root]:
            if search_dir and search_dir not in search_dirs:
                search_dirs.append(search_dir)

        # Tìm transcript trong các thư mục ưu tiên (không đệ quy)
        for search_dir in search_dirs:
            for candidate in transcript_candidates:
                candidate_path = os.path.join(search_dir, candidate)
                if candidate_path not in transcript_search_paths:
                    transcript_search_paths.append(candidate_path)
                if os.path.exists(candidate_path):
                    found_transcript = candidate_path
                    break
            if found_transcript:
                break

        # Nếu chưa tìm thấy, duyệt toàn bộ thư mục con
        if not found_transcript:
            for search_dir in search_dirs:
                for root, _, files in os.walk(search_dir):
                    for candidate in transcript_candidates:
                        if candidate in files:
                            found_transcript = os.path.join(root, candidate)
                            transcript_search_paths.append(found_transcript)
                            break
                    if found_transcript:
                        break
                if found_transcript:
                    break

        if found_transcript:
            prompts_file = os.path.abspath(found_transcript)
            print(f"✓ Tìm thấy file transcript: {prompts_file}")
        else:
            fallback_root = None
            for root_candidate in [archive_super_root, archive_root, archive_dir]:
                if root_candidate:
                    fallback_root = root_candidate
                    break
            prompts_file = os.path.join(fallback_root, 'prompts.txt') if fallback_root else 'prompts.txt'
            if prompts_file not in transcript_search_paths:
                transcript_search_paths.append(prompts_file)
    
    # Xác định thư mục audio
    audio_search_paths = []
    if args.audio_dir:
        audio_dir = os.path.abspath(args.audio_dir)
        audio_search_paths.append(audio_dir)
    else:
        audio_candidates = [
            'mp3',
            'mp3s',
            'wav',
            'wavs',
            'waves',
            'audio',
        ]
        found_audio_dir = None
        audio_search_dirs = []
        for search_dir in [archive_dir, archive_root, archive_super_root]:
            if search_dir and search_dir not in audio_search_dirs:
                audio_search_dirs.append(search_dir)
        # Ưu tiên tìm trực tiếp trong các thư mục ưu tiên
        for search_dir in audio_search_dirs:
            for candidate in audio_candidates:
                candidate_path = os.path.join(search_dir, candidate)
                if candidate_path not in audio_search_paths:
                    audio_search_paths.append(candidate_path)
                if os.path.isdir(candidate_path):
                    found_audio_dir = candidate_path
                    break
            if found_audio_dir:
                break
        # Nếu chưa thấy, duyệt đệ quy
        if not found_audio_dir:
            for search_dir in audio_search_dirs:
                for root, dirs, _ in os.walk(search_dir):
                    for candidate in audio_candidates:
                        if candidate in dirs:
                            found_audio_dir = os.path.join(root, candidate)
                            audio_search_paths.append(found_audio_dir)
                            break
                    if found_audio_dir:
                        break
                if found_audio_dir:
                    break
        if found_audio_dir:
            audio_dir = os.path.abspath(found_audio_dir)
            print(f"✓ Tìm thấy thư mục audio: {audio_dir}")
        else:
            fallback_root = None
            for root_candidate in [archive_super_root, archive_root, archive_dir]:
                if root_candidate:
                    fallback_root = root_candidate
                    break
            audio_dir = os.path.join(fallback_root, 'mp3') if fallback_root else 'mp3'
            if audio_dir not in audio_search_paths:
                audio_search_paths.append(audio_dir)
    
    if not os.path.exists(prompts_file):
        print(f"Error: File transcript không tồn tại: {prompts_file}")
        if transcript_search_paths:
            print("  Đã thử tìm trong:")
            for path in transcript_search_paths:
                print(f"    - {path}")
        print("  Hãy chỉ định --transcript-file hoặc đảm bảo có file transcript hợp lệ trong archive-dir")
        return
    
    if not os.path.exists(audio_dir):
        print(f"Error: Thư mục audio không tồn tại: {audio_dir}")
        if audio_search_paths:
            print("  Đã thử tìm trong:")
            for path in audio_search_paths:
                print(f"    - {path}")
        print("  Hãy chỉ định --audio-dir hoặc đảm bảo có thư mục chứa audio trong archive-dir")
        return
    
    # Tạo output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tạo dataset entries (bao gồm cả positive và negative)
    # audio_dir sẽ được dùng làm base path trong JSONL (relative paths)
    entries = create_dataset_entries(
        audio_dir, 
        prompts_file, 
        negative_ratio=args.negative_ratio,
        seed=args.seed
    )
    
    if len(entries) == 0:
        print("Error: Không có entries nào để chia")
        return
    
    # Lấy mẫu dataset nếu được yêu cầu (ví dụ: chỉ lấy 10% dataset gốc)
    if dataset_ratio < 1.0:
        entries = sample_dataset(entries, ratio=dataset_ratio, seed=args.seed)
    
    if len(entries) == 0:
        print("Error: Không có entries nào sau khi lấy mẫu")
        return
    
    # Chia dataset (đảm bảo mỗi tập đều có cả positive và negative)
    train_entries, test_entries, dev_entries = split_dataset(
        entries,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed
    )
    
    # Ghi các file JSONL
    train_file = os.path.join(args.output_dir, 'train.jsonl')
    test_file = os.path.join(args.output_dir, 'test.jsonl')
    dev_file = os.path.join(args.output_dir, 'dev.jsonl')
    
    write_jsonl(train_entries, train_file)
    write_jsonl(test_entries, test_file)
    write_jsonl(dev_entries, dev_file)
    
    print(f"\n✅ Hoàn thành! Đã tạo 3 file JSONL trong thư mục: {args.output_dir}")
    print(f"  - {train_file}")
    print(f"  - {test_file}")
    print(f"  - {dev_file}")
    print(f"\n📊 Mỗi file JSONL đều chứa cả positive (is_match=true) và negative (is_match=false) samples")


if __name__ == '__main__':
    main()


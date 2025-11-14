#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script cải thiện dataset:
1. Normalize text (loại bỏ ký tự đặc biệt)
2. Loại bỏ hoặc xử lý negative samples
3. Xử lý câu quá ngắn/dài
"""

import json
import argparse
import re
from pathlib import Path


def normalize_text(text):
    """
    Normalize text: loại bỏ ký tự đặc biệt, normalize khoảng trắng
    """
    if not text:
        return ""
    
    # Loại bỏ ký tự đặc biệt
    text = text.replace('\\r\\n', ' ').replace('\\r', ' ').replace('\\n', ' ')
    text = text.replace('\\t', ' ').replace('\t', ' ')
    
    # Loại bỏ khoảng trắng thừa
    text = ' '.join(text.split())
    
    return text.strip()


def should_keep_sample(sentence, min_words=2, max_words=60):
    """
    Kiểm tra xem sample có nên được giữ lại không
    """
    if not sentence or not sentence.strip():
        return False, "Câu trống"
    
    word_count = len(sentence.split())
    
    if word_count < min_words:
        return False, f"Câu quá ngắn ({word_count} từ)"
    
    if word_count > max_words:
        return False, f"Câu quá dài ({word_count} từ)"
    
    return True, None


def improve_dataset(input_jsonl, output_jsonl, 
                   remove_negative=False, 
                   normalize=True,
                   min_words=2,
                   max_words=60,
                   stats_only=False):
    """
    Cải thiện dataset
    """
    print(f"Đang đọc dataset từ: {input_jsonl}")
    
    data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Bỏ qua dòng không hợp lệ: {e}")
                continue
    
    print(f"Tổng số mẫu ban đầu: {len(data):,}")
    
    # Thống kê
    stats = {
        'total': len(data),
        'positive': 0,
        'negative': 0,
        'removed_negative': 0,
        'removed_short': 0,
        'removed_long': 0,
        'removed_empty': 0,
        'normalized': 0,
        'final': 0
    }
    
    improved_data = []
    
    for item in data:
        original_sentence = item.get('sentence', '')
        is_match = item.get('is_match', True)
        
        # Thống kê
        if is_match:
            stats['positive'] += 1
        else:
            stats['negative'] += 1
        
        # Loại bỏ negative samples nếu được yêu cầu
        if remove_negative and not is_match:
            stats['removed_negative'] += 1
            continue
        
        # Normalize text
        if normalize:
            normalized_sentence = normalize_text(original_sentence)
            if normalized_sentence != original_sentence:
                stats['normalized'] += 1
            sentence = normalized_sentence
        else:
            sentence = original_sentence
        
        # Kiểm tra độ dài
        keep, reason = should_keep_sample(sentence, min_words, max_words)
        if not keep:
            if reason.startswith("Câu trống"):
                stats['removed_empty'] += 1
            elif reason.startswith("Câu quá ngắn"):
                stats['removed_short'] += 1
            elif reason.startswith("Câu quá dài"):
                stats['removed_long'] += 1
            continue
        
        # Tạo item mới với text đã normalize
        new_item = item.copy()
        new_item['sentence'] = sentence
        improved_data.append(new_item)
    
    stats['final'] = len(improved_data)
    
    # In thống kê
    print(f"\n{'='*70}")
    print(f"📊 THỐNG KÊ CẢI THIỆN DATASET")
    print(f"{'='*70}")
    print(f"Tổng số mẫu ban đầu: {stats['total']:,}")
    print(f"  - Positive: {stats['positive']:,}")
    print(f"  - Negative: {stats['negative']:,}")
    print(f"\nĐã loại bỏ:")
    print(f"  - Negative samples: {stats['removed_negative']:,}")
    print(f"  - Câu trống: {stats['removed_empty']:,}")
    print(f"  - Câu quá ngắn (<{min_words} từ): {stats['removed_short']:,}")
    print(f"  - Câu quá dài (>{max_words} từ): {stats['removed_long']:,}")
    print(f"\nĐã normalize: {stats['normalized']:,} câu")
    print(f"\n✅ Số mẫu cuối cùng: {stats['final']:,} ({stats['final']/stats['total']*100:.1f}%)")
    print(f"{'='*70}\n")
    
    if stats_only:
        return stats
    
    # Lưu dataset đã cải thiện
    if output_jsonl:
        print(f"Đang lưu dataset đã cải thiện vào: {output_jsonl}")
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for item in improved_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Đã lưu {len(improved_data):,} mẫu vào {output_jsonl}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Cải thiện dataset: normalize text, loại bỏ negative samples, xử lý câu quá ngắn/dài'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='File JSONL input (ví dụ: data/train.jsonl)')
    parser.add_argument('--output', type=str, default=None,
                       help='File JSONL output (ví dụ: data/train_improved.jsonl). Nếu không chỉ định, sẽ in thống kê mà không lưu')
    parser.add_argument('--remove-negative', action='store_true',
                       help='Loại bỏ negative samples (is_match=False)')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false', default=True,
                       help='Tắt normalize text')
    parser.add_argument('--min-words', type=int, default=2,
                       help='Số từ tối thiểu (mặc định: 2)')
    parser.add_argument('--max-words', type=int, default=60,
                       help='Số từ tối đa (mặc định: 60)')
    parser.add_argument('--stats-only', action='store_true',
                       help='Chỉ in thống kê, không lưu file')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"❌ File không tồn tại: {args.input}")
        return
    
    improve_dataset(
        input_jsonl=args.input,
        output_jsonl=None if args.stats_only else (args.output or args.input.replace('.jsonl', '_improved.jsonl')),
        remove_negative=args.remove_negative,
        normalize=args.normalize,
        min_words=args.min_words,
        max_words=args.max_words,
        stats_only=args.stats_only
    )


if __name__ == "__main__":
    main()


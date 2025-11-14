"""
Script để tải PhoWhisper-large model từ HuggingFace trước khi sử dụng
Model sẽ được lưu vào cache của HuggingFace để sử dụng sau này
"""

import os
from transformers import pipeline
import torch

def download_phowhisper_large():
    """Tải PhoWhisper-large model từ HuggingFace"""
    print("=" * 60)
    print("Đang tải PhoWhisper-large model...")
    print("=" * 60)
    print(f"Model ID: vinai/phowhisper-large")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Kích thước ước tính: ~3-4 GB")
    print("=" * 60)
    print()
    
    try:
        print("Bước 1: Đang tải model và tokenizer...")
        pipe = pipeline(
            "automatic-speech-recognition",
            model="vinai/phowhisper-large",
            device=0 if torch.cuda.is_available() else -1,
        )
        print("✓ Đã tải model thành công!")
        print()
        
        print("Bước 2: Kiểm tra model...")
        print(f"✓ Model đã sẵn sàng sử dụng")
        print(f"✓ Model được lưu tại: {os.path.expanduser('~/.cache/huggingface/hub')}")
        print()
        
        print("=" * 60)
        print("HOÀN TẤT! PhoWhisper-large đã sẵn sàng sử dụng.")
        print("=" * 60)
        print()
        print("Bạn có thể chạy app.py bây giờ, model sẽ không cần tải lại.")
        
    except Exception as e:
        print(f"✗ Lỗi khi tải model: {e}")
        print()
        print("Nguyên nhân có thể:")
        print("1. Không có kết nối internet")
        print("2. Không đủ dung lượng ổ cứng")
        print("3. Thiếu thư viện cần thiết (transformers, torch)")
        print()
        print("Giải pháp:")
        print("1. Kiểm tra kết nối internet")
        print("2. Đảm bảo có ít nhất 5GB dung lượng trống")
        print("3. Chạy: pip install transformers torch")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    download_phowhisper_large()


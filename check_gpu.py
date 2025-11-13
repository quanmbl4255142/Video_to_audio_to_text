"""
Script kiểm tra GPU và dependencies cho fine-tuning
"""

import sys

print("=== Kiểm tra môi trường ===")
print()

# Kiểm tra PyTorch
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  GPU Count: {torch.cuda.device_count()}")
    else:
        print("  ⚠ Warning: Không có GPU. Training sẽ chậm hơn nhiều.")
        print("  Hãy đảm bảo đã cài đặt PyTorch với CUDA support:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
except ImportError:
    print("✗ PyTorch chưa được cài đặt")
    sys.exit(1)

print()

# Kiểm tra librosa
try:
    import librosa
    print(f"✓ librosa: {librosa.__version__}")
except ImportError:
    print("✗ librosa chưa được cài đặt")
    print("  Cài đặt: pip install librosa")
    sys.exit(1)

# Kiểm tra soundfile
try:
    import soundfile
    print(f"✓ soundfile: {soundfile.__version__}")
except ImportError:
    print("✗ soundfile chưa được cài đặt")
    print("  Cài đặt: pip install soundfile")
    sys.exit(1)

# Kiểm tra transformers
try:
    import transformers
    print(f"✓ transformers: {transformers.__version__}")
except ImportError:
    print("✗ transformers chưa được cài đặt")
    sys.exit(1)

# Kiểm tra datasets
try:
    import datasets
    print(f"✓ datasets: {datasets.__version__}")
except ImportError:
    print("✗ datasets chưa được cài đặt")
    sys.exit(1)

print()
print("=== Tất cả dependencies đã sẵn sàng! ===")
if torch.cuda.is_available():
    print("✓ GPU đã được phát hiện và sẵn sàng sử dụng")
else:
    print("⚠ Không có GPU - training sẽ chạy trên CPU (rất chậm)")


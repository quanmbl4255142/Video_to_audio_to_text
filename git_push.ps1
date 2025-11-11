# Script để commit và push code lên GitHub
# Chạy trong PowerShell: .\git_push.ps1

Write-Host "=== Git Commit và Push ===" -ForegroundColor Green

# Kiểm tra xem đã có remote chưa
Write-Host "`n1. Kiểm tra remote..." -ForegroundColor Yellow
git remote -v

# Kiểm tra branch hiện tại
Write-Host "`n2. Kiểm tra branch..." -ForegroundColor Yellow
git branch

# Add tất cả files (trừ venv đã được ignore)
Write-Host "`n3. Đang add files..." -ForegroundColor Yellow
git add .

# Kiểm tra status
Write-Host "`n4. Kiểm tra files đã add..." -ForegroundColor Yellow
git status --short | Select-Object -First 20

# Commit
Write-Host "`n5. Đang commit..." -ForegroundColor Yellow
$commitMessage = "Initial commit: Video to Audio Text project with PhoWhisper fine-tuning"
git commit -m $commitMessage

# Kiểm tra xem có branch main chưa, nếu chưa thì tạo
Write-Host "`n6. Kiểm tra và tạo branch main nếu cần..." -ForegroundColor Yellow
$currentBranch = git branch --show-current
if ($currentBranch -ne "main") {
    Write-Host "Đang đổi tên branch từ $currentBranch sang main..." -ForegroundColor Cyan
    git branch -M main
}

# Push lên GitHub
Write-Host "`n7. Đang push lên GitHub..." -ForegroundColor Yellow
git push -u origin main

Write-Host "`n=== Hoàn thành! ===" -ForegroundColor Green
Write-Host "Kiểm tra tại: https://github.com/quanmbl4255142/Video_to_audio_to_text" -ForegroundColor Cyan


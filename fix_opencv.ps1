# Script to fix OpenCV DLL issues on Windows
Write-Host "Fixing OpenCV installation..." -ForegroundColor Cyan

# Check Python version
$pythonVersion = python --version 2>&1
Write-Host "Python version: $pythonVersion"

# Uninstall existing opencv packages
Write-Host "`nUninstalling existing OpenCV packages..." -ForegroundColor Yellow
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y 2>&1 | Out-Null

# Try installing with specific flags
Write-Host "`nInstalling opencv-python..." -ForegroundColor Yellow
pip install --upgrade --force-reinstall --no-cache-dir opencv-python

# Test import
Write-Host "`nTesting OpenCV import..." -ForegroundColor Yellow
$testResult = python -c "import cv2; print('SUCCESS: OpenCV version', cv2.__version__)" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] OpenCV is working!" -ForegroundColor Green
    Write-Host $testResult
} else {
    Write-Host "[ERROR] OpenCV still not working." -ForegroundColor Red
    Write-Host $testResult
    Write-Host "`nPossible solutions:" -ForegroundColor Yellow
    Write-Host "1. Install Microsoft Visual C++ Redistributables:" -ForegroundColor White
    Write-Host "   Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor Gray
    Write-Host "2. Try using Python 3.11 or 3.12 instead of 3.13" -ForegroundColor White
    Write-Host "3. Try: pip install opencv-python-headless" -ForegroundColor White
}

# PowerShell script to download pretrained models
$pretrainedDir = "../pretrained_models"

# Create directory if it doesn't exist
if (-not (Test-Path $pretrainedDir)) {
    New-Item -ItemType Directory -Path $pretrainedDir -Force | Out-Null
}

# Function to download with retry
function Download-FileWithRetry {
    param(
        [string]$Uri,
        [string]$OutFile,
        [int]$MaxRetries = 3,
        [int]$TimeoutSec = 300
    )
    
    $retryCount = 0
    while ($retryCount -lt $MaxRetries) {
        try {
            Write-Host "Downloading $OutFile (attempt $($retryCount + 1)/$MaxRetries)..."
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -Uri $Uri -OutFile $OutFile -TimeoutSec $TimeoutSec -UseBasicParsing
            Write-Host "[SUCCESS] Successfully downloaded $OutFile"
            return $true
        }
        catch {
            $retryCount++
            if ($retryCount -lt $MaxRetries) {
                Write-Host "[RETRY] Download failed. Retrying in 5 seconds..." -ForegroundColor Yellow
                Start-Sleep -Seconds 5
            }
            else {
                Write-Host "[ERROR] Failed to download $OutFile after $MaxRetries attempts: $_" -ForegroundColor Red
                return $false
            }
        }
    }
}

# Download files
$success = $true

$success = (Download-FileWithRetry -Uri "https://github.com/LeslieZhoa/HeSer.Pytorch/releases/download/v0.0/parsing.pth" -OutFile "$pretrainedDir/parsing.pth") -and $success

$success = (Download-FileWithRetry -Uri "https://github.com/LeslieZhoa/HeadSwap/releases/download/v0.0/sr_cf.onnx" -OutFile "$pretrainedDir/sr_cf.onnx") -and $success

$success = (Download-FileWithRetry -Uri "https://github.com/LeslieZhoa/HeadSwap/releases/download/v0.0/Blender-401-00012900.pth" -OutFile "$pretrainedDir/Blender-401-00012900.pth") -and $success

if ($success) {
    Write-Host ""
    Write-Host "[SUCCESS] All downloads complete!" -ForegroundColor Green
}
else {
    Write-Host ""
    Write-Host "[WARNING] Some downloads failed. Please check the errors above and retry if needed." -ForegroundColor Yellow
}

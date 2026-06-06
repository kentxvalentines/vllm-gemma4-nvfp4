# Local model-baked image builder for vllm-gemma4-nvfp4
$ErrorActionPreference = 'Stop'

$ModelFolder = 'D:\Gits\text-generation-webui\user_data\models\Huihui-gemma-4-26B-A4B-it-abliterated-NVFP4'
$ConfigsToStage = @('chat_template.jinja')

if (-not (Test-Path $ModelFolder)) {
    Write-Host 'ERROR: Model folder not found at '$ModelFolder'!' -ForegroundColor Red
    exit 1
}

# Stage config files temporarily in the build context (the model folder)
$StagedFiles = @()
foreach ($Cfg in $ConfigsToStage) {
    $Src = Join-Path $PSScriptRoot $Cfg
    $Dest = Join-Path $ModelFolder $Cfg
    if (Test-Path $Src) {
        Copy-Item -Path $Src -Destination $Dest -Force
        $StagedFiles += $Dest
    }
}

Write-Host 'Building Docker image using standard build context...' -ForegroundColor Cyan
$BuildArgs = if ($env:HF_TOKEN) { "--build-arg HF_TOKEN=$($env:HF_TOKEN)" } else { "" }
$BuildCommand = "docker build $BuildArgs -f Dockerfile -t vllm-gemma4-nvfp4:latest \"$ModelFolder\""
Write-Host 'Command: '$BuildCommand -ForegroundColor DarkGray
Write-Host

try {
    Invoke-Expression $BuildCommand
    Write-Host '
SUCCESS: vllm-gemma4-nvfp4 built successfully!' -ForegroundColor Green
} catch {
    Write-Host '
FAILURE: Failed to build vllm-gemma4-nvfp4.' -ForegroundColor Red
    Write-Host $.Exception.Message -ForegroundColor Red
} finally {
    # Clean up
    if ($StagedFiles.Count -gt 0) {
        Write-Host 'Cleaning up staged configurations...' -ForegroundColor Gray
        foreach ($StagedFile in $StagedFiles) {
            if (Test-Path $StagedFile) {
                Remove-Item -Path $StagedFile -Force
            }
        }
    }
}

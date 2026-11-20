<#
.SYNOPSIS
    1-Click Launcher for Qwen3.8-9B Recurrent-Loop (1..6 Loops) with llama-server Web UI.
.DESCRIPTION
    Automatically downloads the base GGUF (mradermacher/Qwen3.8-9B-Distill-uncensored-heretic-GGUF)
    and the Loop-6 adapter (Vnswz/Qwen3.8-9B-Loop6-Adapter) if missing, then starts llama-server
    with presets for 1, 2, 4, and 6 loops and opens the Web UI at http://localhost:8080.
#>

param(
    [int]$Port = 8080,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# Locate llama-server.exe (release zip root OR local CMake build folder)
$ServerExe = Join-Path $Root "llama-server.exe"
if (-not (Test-Path $ServerExe)) {
    $ServerExe = Join-Path $Root "build\bin\Release\llama-server.exe"
}
if (-not (Test-Path $ServerExe)) {
    Write-Host "ERROR: llama-server.exe not found in current folder or build\bin\Release\." -ForegroundColor Red
    Write-Host "Download the prebuilt Windows CUDA release ZIP or build with CMake first." -ForegroundColor Yellow
    exit 1
}

$ModelsDir = Join-Path $Root "models"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

$BaseFile = Join-Path $ModelsDir "Qwen3.8-9B-Distill-uncensored-heretic.Q4_K_S.gguf"
$BaseUrl  = "https://huggingface.co/mradermacher/Qwen3.8-9B-Distill-uncensored-heretic-GGUF/resolve/main/Qwen3.8-9B-Distill-uncensored-heretic.Q4_K_S.gguf"

$LoraFile = Join-Path $ModelsDir "qwen3.8-9b-heretic-loop6-lora.gguf"
$LoraUrl  = "https://huggingface.co/Vnswz/Qwen3.8-9B-Loop6-Adapter/resolve/main/qwen3.8-9b-heretic-loop6-lora.gguf"

function Download-IfMissing([string]$Dest, [string]$Url, [string]$Label) {
    if (Test-Path $Dest) {
        Write-Host "[OK] Found $Label -> $Dest" -ForegroundColor Green
        return
    }
    Write-Host "[Download] Fetching $Label ..." -ForegroundColor Cyan
    Write-Host "  From: $Url" -ForegroundColor DarkGray
    $Tmp = "$Dest.part"
    curl.exe -L --fail --retry 3 --progress-bar -o $Tmp $Url
    if ($LASTEXITCODE -ne 0) {
        if (Test-Path $Tmp) { Remove-Item -Force $Tmp }
        Write-Host "ERROR: Failed to download $Label." -ForegroundColor Red
        Write-Host "If the Hugging Face repository is private, download it manually and place it at: $Dest" -ForegroundColor Yellow
        exit 1
    }
    Move-Item -Force $Tmp $Dest
    Write-Host "[OK] Saved $Label" -ForegroundColor Green
}

Download-IfMissing $BaseFile $BaseUrl "Base Model (Qwen3.8-9B Q4_K_S, ~5.23 GB)"
Download-IfMissing $LoraFile $LoraUrl "Loop-6 Adapter (~45 MB)"

$PresetFile = Join-Path $Root "models.example.ini"
if (-not (Test-Path $PresetFile)) {
    Write-Host "ERROR: models.example.ini not found." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Starting Qwen3.8-9B Recurrent-Loop Server on http://localhost:$Port" -ForegroundColor Cyan
Write-Host " Switch between 1-Loop, 2-Loop, 4-Loop, and 6-Loop in the Web UI!" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($p)
        Start-Sleep -Seconds 3
        Start-Process "http://localhost:$p"
    } -ArgumentList $Port | Out-Null
}

& $ServerExe --models-preset $PresetFile --port $Port

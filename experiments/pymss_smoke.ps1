# pymss 最小分离冒烟脚本（M0.7）
# 在 experiments 内创建独立 uv 环境，安装 CPU torch + pymss，下载 40MB SCNet 模型并分离一段音频。
# 只写 experiments/ 与 07_执行计划/验证/audio/；不安装系统级软件；不修改源目录。
#
# 用法：
#   pwsh -File pymss_smoke.ps1                # CPU（默认；约 2-4 分钟，CPU torch 206MiB）
#   pwsh -File pymss_smoke.ps1 -Gpu           # CUDA 12.8（torch wheel 3.05GiB，需较长时间）
#   pwsh -File pymss_smoke.ps1 -InputAudio "D:\x.wav" -OutputDir "D:\out"

param(
    [string]$InputAudio = (Join-Path $PSScriptRoot "work\input_music_20s.wav"),
    [string]$OutputDir = "F:\Deepseek_harness\AI音乐处理\07_执行计划\验证\audio\pymss_scnet_20s",
    [switch]$Gpu,
    [string]$Model = "scnet_checkpoint_musdb18"
)

$ErrorActionPreference = "Stop"
$exp = $PSScriptRoot
$uvEnv = Join-Path $exp "tools-uv\.venv"
$pymssEnv = Join-Path $exp "pymss\.venv"
$models = Join-Path $exp "pymss\models"

$env:UV_CACHE_DIR = Join-Path $exp "cache\uv"
$env:TMP = Join-Path $exp "tmp"
$env:TEMP = $env:TMP
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYMSS_MODEL_DIR = $models

if (-not (Test-Path $InputAudio)) { throw "输入音频不存在：$InputAudio" }
New-Item -ItemType Directory -Force -Path $models | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# 1) 本地 uv（仅本目录，不装系统级）
$systemPy = (Get-Command python -ErrorAction Stop).Source
if (-not (Test-Path (Join-Path $uvEnv "Scripts\uv.exe"))) {
    Write-Host "[1/5] 创建本地 uv 工具环境 ..."
    & $systemPy -m venv $uvEnv
    & (Join-Path $uvEnv "Scripts\python.exe") -m pip install --disable-pip-version-check -q uv
}
$uv = Join-Path $uvEnv "Scripts\uv.exe"

# 2) pymss venv（Python 3.12）
if (-not (Test-Path (Join-Path $pymssEnv "Scripts\python.exe"))) {
    Write-Host "[2/5] 创建 pymss 独立环境（Python 3.12） ..."
    & $uv venv $pymssEnv --python $systemPy
}
$py = Join-Path $pymssEnv "Scripts\python.exe"

# 3) torch + pymss
$torchIndex = if ($Gpu) { "https://download.pytorch.org/whl/cu128" } else { "https://download.pytorch.org/whl/cpu" }
$torchLabel = if ($Gpu) { "cu128（约 3.05GiB）" } else { "cpu（约 206MiB）" }
Write-Host "[3/5] 安装 torch 2.7.1 $torchLabel + pymss ..."
& $uv pip install --python $py --index-url $torchIndex "torch==2.7.1"
& $uv pip install --python $py "pymss" "psutil"
& $py -c "import torch, pymss; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 4) 下载 4 轨小模型
Write-Host "[4/5] 下载模型 $Model（约 42MB） ..."
& $py -m pymss.cli download $Model --model-dir $models --source hf-mirror

# 5) 分离
$device = if ($Gpu) { "cuda" } else { "cpu" }
Write-Host "[5/5] 分离（device=$device） ..."
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $py -m pymss.cli infer $Model -i $InputAudio -o $OutputDir --save-as-folder --device $device --format wav --wav-bit-depth FLOAT --model-dir $models --source hf-mirror --param batch_size=1
$sw.Stop()
Write-Host ("完成：$OutputDir（耗时 " + [math]::Round($sw.Elapsed.TotalSeconds,1) + "s）")
Get-ChildItem $OutputDir -Recurse -File | Select-Object Name, Length | Format-Table -AutoSize

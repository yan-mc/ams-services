# Gateway MVP 冒烟脚本（Windows PowerShell 7）
# 启动真实 uvicorn -> health -> 创建 ping 任务 -> 查询产物 -> 打印 SSE -> 停止服务。
# 只写 gateway/.ams_home_dev（已 gitignore），不碰用户数据。

param(
    [string]$AmsHome = (Join-Path (Split-Path $PSScriptRoot -Parent) ".ams_home_dev"),
    [int]$Port = 0,
    [int]$TimeoutSec = 40
)

$ErrorActionPreference = "Stop"
$gatewayDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $gatewayDir ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "未找到 .venv\Scripts\python.exe，请先执行：python -m venv .venv && .venv\Scripts\python.exe -m pip install -e '.[dev]'"
}
New-Item -ItemType Directory -Force -Path $AmsHome | Out-Null
$stdout = Join-Path $AmsHome "smoke-stdout.log"
$stderr = Join-Path $AmsHome "smoke-stderr.log"

$proc = Start-Process -FilePath $python -ArgumentList @(
    "-m", "app", "--ams-home", $AmsHome, "--port", $Port, "--log-level", "INFO"
) -WorkingDirectory $gatewayDir -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru

try {
    $portsFile = Join-Path $AmsHome "runtime\ports.json"
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while (-not (Test-Path $portsFile) -and (Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 200
        if ($proc.HasExited) { break }
    }
    if (-not (Test-Path $portsFile)) {
        Get-Content $stdout -ErrorAction SilentlyContinue | Select-Object -Last 30
        Get-Content $stderr -ErrorAction SilentlyContinue | Select-Object -Last 30
        throw "ports.json 未生成，服务可能启动失败"
    }
    $ports = Get-Content -Raw $portsFile | ConvertFrom-Json
    $base = "http://127.0.0.1:$($ports.gateway.port)/api/v1"
    $token = (Get-Content -Raw $ports.gateway.token_file).Trim()
    $headers = @{ Authorization = "Bearer $token" }

    $health = Invoke-RestMethod -Uri "$base/health" -Headers $headers
    Write-Host "[1] health ok：version=$($health.version) 端口=$($ports.gateway.port) 队列=$($health.db.jobs | ConvertTo-Json -Compress)"

    $job = Invoke-RestMethod -Uri "$base/jobs" -Method Post -Headers $headers -ContentType "application/json" -Body '{"type":"ping","params":{}}'
    Write-Host "[2] 创建任务：$($job.id) status=$($job.status)"

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    do {
        Start-Sleep -Milliseconds 150
        $current = Invoke-RestMethod -Uri "$base/jobs/$($job.id)" -Headers $headers
    } while ($current.status -in @("queued", "running") -and (Get-Date) -lt $deadline)
    Write-Host "[3] 任务完成：status=$($current.status) progress=$($current.progress)"

    $artifacts = Invoke-RestMethod -Uri "$base/jobs/$($job.id)/artifacts" -Headers $headers
    Write-Host "[4] 产物数=$($artifacts.data.Count) 首个=$($artifacts.data[0].name) sha256=$($artifacts.data[0].sha256.Substring(0,12))..."

    Write-Host "[5] SSE 事件（前 12 行）："
    & curl.exe -s -N --max-time 5 -H "Authorization: Bearer $token" "$base/jobs/$($job.id)/events" |
        Select-Object -First 12

    Write-Host "[6] 冒烟通过 ✔"
}
finally {
    if ($proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Write-Host "[7] 已停止 Gateway 进程 pid=$($proc.Id)"
    }
}

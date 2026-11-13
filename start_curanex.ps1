# Start the whole Curanex app on this computer (after a restart too). Right-click -> "Run with PowerShell".
# Starts Docker Desktop + image workers, the API, the tabular worker and the web app, waits until each answers, then opens the browser.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$backend = Join-Path $root "backend"
$py = Join-Path $backend ".venv\Scripts\python.exe"
$node = "$env:USERPROFILE\tools\node-v22.14.0-win-x64\node.exe"
if (-not (Test-Path $node)) { $node = (Get-Command node).Source }

function Wait-Url($url, $name, $seconds) {
    $end = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $end) {
        try { Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5 | Out-Null; Write-Host "  OK  $name" -ForegroundColor Green; return } catch { Start-Sleep -Seconds 3 }
    }
    Write-Host "  NOT READY  $name ($url)" -ForegroundColor Red
}

Write-Host "1/4 Docker Desktop and image workers"
docker info 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Start-Process "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe"
    do { Start-Sleep -Seconds 3; docker info 2>$null | Out-Null } while ($LASTEXITCODE -ne 0)
}
docker start backend-worker-torch-1 backend-worker-tf-1 curanex-postgres | Out-Null

Write-Host "2/4 API and tabular worker"
Start-Process -WindowStyle Minimized $py -ArgumentList "-m uvicorn main:app --app-dir `"$backend`" --host 0.0.0.0 --port 8001"
Start-Process -WindowStyle Minimized $py -ArgumentList "-m uvicorn workers.common.server:app --app-dir `"$backend`" --host 127.0.0.1 --port 8101"

Write-Host "3/4 Web app"
Start-Process -WindowStyle Minimized $node -ArgumentList "`"$root\node_modules\expo\bin\cli`" start --web --port 8081 `"$root`"" -WorkingDirectory $root

Write-Host "4/4 Waiting for everything to answer"
Wait-Url "http://localhost:8001/health" "API (8001)" 120
Wait-Url "http://localhost:8101/health" "Tabular models (8101)" 120
Wait-Url "http://localhost:8102/health" "Image models, torch (8102)" 180
Wait-Url "http://localhost:8103/health" "Image models, tf (8103)" 180
Wait-Url "http://localhost:8081" "Web app (8081)" 240
Start-Process "http://localhost:8081"
$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.PrefixOrigin -ne "WellKnown" } | Select-Object -First 1).IPAddress
Write-Host "Curanex is running. On this PC: http://localhost:8081   Doctor Desk: /doctor-portal   Admin: /admin"
Write-Host "On your iPhone (same Wi-Fi or this PC on the iPhone hotspot): open Safari at http://${ip}:8081" -ForegroundColor Cyan

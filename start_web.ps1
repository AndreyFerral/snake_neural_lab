param(
    [string]$BackendHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$ForceCpu,
    [switch]$AllowGpu,
    [switch]$NoBrowser,
    [switch]$Detached,
    [switch]$Restart
)

$ErrorActionPreference = "Stop"

# ---- Определение корня проекта ----
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Find-ProjectRoot {
    param([string]$StartDir)
    $current = Resolve-Path $StartDir
    while ($current.Drive) {
        if (Test-Path (Join-Path $current "web_backend.py")) {
            return $current.ToString()
        }
        $parent = Split-Path $current -Parent
        if ($parent -eq $current) { break }
        $current = $parent
    }
    throw "Could not find web_backend.py in any parent directory of $StartDir"
}

$ProjectRoot = Find-ProjectRoot -StartDir $scriptDir
$BackendScript = Join-Path $ProjectRoot "web_backend.py"
$RuntimeDir = Join-Path $ProjectRoot ".runtime"
$PidFile = Join-Path $RuntimeDir "web_backend.pid"
$RestartFlagFile = Join-Path $RuntimeDir "restart_requested.flag"
$Url = "http://${BackendHost}:${Port}"
$HealthUrl = "${Url}/api/health"
$CommandUrl = "${Url}/api/command"

# ---- Кеш Python ----
$script:PythonExecutable = $null
$script:UsePyLauncher = $false

# ---- Функции ----
function Test-BackendHealthy {
    param([string]$TargetUrl)
    try {
        $response = Invoke-WebRequest -Uri $TargetUrl -UseBasicParsing -TimeoutSec 1
        return $response.StatusCode -eq 200
    } catch { return $false }
}

function Get-PythonLaunch {
    param(
        [string]$ScriptPath,
        [string]$BackendHost,
        [int]$BackendPort,
        [bool]$UseForceCpu,
        [bool]$UseAllowGpu
    )
    $arguments = @($ScriptPath, "--host", $BackendHost, "--port", "$BackendPort")
    if ($UseForceCpu) { $arguments += "--force-cpu" }
    if ($UseAllowGpu) { $arguments += "--allow-gpu" }

    if (-not $script:PythonExecutable) {
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCmd) {
            $script:PythonExecutable = $pythonCmd.Source
            $script:UsePyLauncher = $false
        } else {
            $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
            if ($pyLauncher) {
                $script:PythonExecutable = $pyLauncher.Source
                $script:UsePyLauncher = $true
            } else {
                throw "Python not found. Install Python or add to PATH."
            }
        }
    }

    $finalArguments = if ($script:UsePyLauncher) { @("-3") + $arguments } else { $arguments }
    return @{ FilePath = $script:PythonExecutable; Arguments = $finalArguments }
}

function Wait-BackendHealthy {
    param([string]$TargetUrl, [int]$Attempts = 30, [int]$DelayMilliseconds = 200)
    for ($i = 0; $i -lt $Attempts; $i++) {
        Start-Sleep -Milliseconds $DelayMilliseconds
        if (Test-BackendHealthy -TargetUrl $TargetUrl) { return $true }
    }
    return $false
}

function Request-BackendRestart { param([string]$TargetUrl)
    $body = @{ type = "restart_server" } | ConvertTo-Json -Compress
    Invoke-WebRequest -Uri $TargetUrl -Method Post -Body $body -ContentType "application/json" -UseBasicParsing -TimeoutSec 1 | Out-Null
}

function Request-BackendShutdown { param([string]$TargetUrl)
    $body = @{ type = "shutdown_server" } | ConvertTo-Json -Compress
    Invoke-WebRequest -Uri $TargetUrl -Method Post -Body $body -ContentType "application/json" -UseBasicParsing -TimeoutSec 1 | Out-Null
}

function Stop-BackendFromPidFile { param([string]$Path)
    if (-not (Test-Path $Path)) { return $false }
    $rawPid = Get-Content -Path $Path -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($rawPid -notmatch '^\d+$') { return $false }
    $processId = [int]$rawPid
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if (-not $process) {
        Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
        return $false
    }
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    return $true
}

# ---- Основная логика ----
if (-not (Test-Path $BackendScript)) {
    throw "Backend script not found: $BackendScript"
}

if ($Restart -and (Test-BackendHealthy -TargetUrl $HealthUrl)) {
    Write-Host "Stopping existing backend at $Url"
    try { Request-BackendShutdown -TargetUrl $CommandUrl }
    catch { Stop-BackendFromPidFile -Path $PidFile | Out-Null }

    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 250
        if (-not (Test-BackendHealthy -TargetUrl $HealthUrl)) { break }
    }
    if (Test-BackendHealthy -TargetUrl $HealthUrl) {
        Stop-BackendFromPidFile -Path $PidFile | Out-Null
        Start-Sleep -Milliseconds 200
    }
    Write-Host "Starting a fresh backend in this console."
}
elseif (Test-BackendHealthy -TargetUrl $HealthUrl) {
    Write-Host "Backend already running at $Url"
    if (-not $NoBrowser) { Start-Process $Url | Out-Null }
    return
}

$launch = Get-PythonLaunch -ScriptPath $BackendScript -BackendHost $BackendHost -BackendPort $Port -UseForceCpu $ForceCpu.IsPresent -UseAllowGpu $AllowGpu.IsPresent
Write-Host "Starting backend: $($launch.FilePath) $($launch.Arguments -join ' ')"

if ($Detached) {
    if (-not (Test-Path $RuntimeDir)) { New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null }
    $process = Start-Process -FilePath $launch.FilePath -ArgumentList $launch.Arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru
    Set-Content -Path $PidFile -Value $process.Id -Encoding ascii

    if (-not (Wait-BackendHealthy -TargetUrl $HealthUrl)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        if (Test-Path $PidFile) { Remove-Item -LiteralPath $PidFile -Force }
        throw "Backend did not become ready at $HealthUrl"
    }
    Write-Host "Backend is ready at $Url"
    if (-not $NoBrowser) { Start-Process $Url | Out-Null }
    Write-Host "Backend is running detached. Closing this console will not stop it."
    return
}

if (-not $NoBrowser) {
    $browserScript = {
        param($TargetHealthUrl, $TargetUrl)
        for ($i = 0; $i -lt 40; $i++) {
            Start-Sleep -Milliseconds 300
            try {
                $r = Invoke-WebRequest -Uri $TargetHealthUrl -UseBasicParsing -TimeoutSec 1
                if ($r.StatusCode -eq 200) { Start-Process $TargetUrl | Out-Null; return }
            } catch {}
        }
    }
    Start-Job -ScriptBlock $browserScript -ArgumentList $HealthUrl, $Url | Out-Null
}

Write-Host "Backend will run in this console. Close this console or press Ctrl+C to stop it."
Push-Location $ProjectRoot
try {
    do {
        if (Test-Path $RestartFlagFile) { Remove-Item -LiteralPath $RestartFlagFile -Force -ErrorAction SilentlyContinue }
        $env:SNAKE_RESTART_IN_PARENT = "1"
        & $launch.FilePath @($launch.Arguments)
        $exitCode = $LASTEXITCODE
        Remove-Item Env:SNAKE_RESTART_IN_PARENT -ErrorAction SilentlyContinue
        $restartRequested = ($exitCode -eq 42) -or (Test-Path $RestartFlagFile)
        if ($restartRequested) {
            Remove-Item -LiteralPath $RestartFlagFile -Force -ErrorAction SilentlyContinue
            Write-Host "Backend restart requested. Starting again..."
            Start-Sleep -Milliseconds 300
        }
    } while ($restartRequested)
    if ($exitCode -ne 0) { Write-Host "Backend exited with code $exitCode" }
}
finally {
    Remove-Item Env:SNAKE_RESTART_IN_PARENT -ErrorAction SilentlyContinue
    Pop-Location
}
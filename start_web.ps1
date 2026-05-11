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

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceRoot = Split-Path -Parent $ProjectRoot
$BackendScript = Join-Path $ProjectRoot "web_backend.py"
$PidFile = Join-Path $ProjectRoot ".web_backend.pid"
$RuntimeDir = Join-Path $ProjectRoot ".runtime"
$RestartFlagFile = Join-Path $RuntimeDir "restart_requested.flag"
$Url = [string]::Format("http://{0}:{1}", $BackendHost, $Port)
$HealthUrl = [string]::Format("{0}/api/health", $Url)
$CommandUrl = [string]::Format("{0}/api/command", $Url)

function Test-BackendHealthy {
    param([string]$TargetUrl)

    try {
        $response = Invoke-WebRequest -Uri $TargetUrl -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
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
    if ($UseForceCpu) {
        $arguments += "--force-cpu"
    }
    if ($UseAllowGpu) {
        $arguments += "--allow-gpu"
    }

    $sharedVenv = Join-Path $WorkspaceRoot "rocm-py\\Scripts\\python.exe"
    if (Test-Path $sharedVenv) {
        return @{
            FilePath = $sharedVenv
            Arguments = $arguments
        }
    }

    $projectVenv = Join-Path $ProjectRoot "rocm-py\\Scripts\\python.exe"
    if (Test-Path $projectVenv) {
        return @{
            FilePath = $projectVenv
            Arguments = $arguments
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return @{
            FilePath = $pythonCommand.Source
            Arguments = $arguments
        }
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        return @{
            FilePath = $pyLauncher.Source
            Arguments = @("-3") + $arguments
        }
    }

    throw "Python launcher not found. Install Python or add it to PATH."
}

function Wait-BackendHealthy {
    param(
        [string]$TargetUrl,
        [int]$Attempts = 60,
        [int]$DelayMilliseconds = 500
    )

    for ($attempt = 0; $attempt -lt $Attempts; $attempt++) {
        Start-Sleep -Milliseconds $DelayMilliseconds
        if (Test-BackendHealthy -TargetUrl $TargetUrl) {
            return $true
        }
    }
    return $false
}

function Request-BackendRestart {
    param([string]$TargetUrl)

    $body = @{ type = "restart_server" } | ConvertTo-Json -Compress
    Invoke-WebRequest -Uri $TargetUrl -Method Post -Body $body -ContentType "application/json" -UseBasicParsing -TimeoutSec 3 | Out-Null
}

function Request-BackendShutdown {
    param([string]$TargetUrl)

    $body = @{ type = "shutdown_server" } | ConvertTo-Json -Compress
    Invoke-WebRequest -Uri $TargetUrl -Method Post -Body $body -ContentType "application/json" -UseBasicParsing -TimeoutSec 3 | Out-Null
}

function Stop-BackendFromPidFile {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return $false
    }

    $rawPid = Get-Content -Path $Path -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($rawPid -notmatch "process:(\d+)") {
        return $false
    }

    $processId = [int]$Matches[1]
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if (-not $process) {
        Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
        return $false
    }

    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    return $true
}

if (-not (Test-Path $BackendScript)) {
    throw "Backend script not found: $BackendScript"
}

if ($Restart -and (Test-BackendHealthy -TargetUrl $HealthUrl)) {
    Write-Host "Stopping existing backend at $Url"
    try {
        Request-BackendShutdown -TargetUrl $CommandUrl
    }
    catch {
        Write-Host "Graceful shutdown request failed; trying PID file fallback."
        Stop-BackendFromPidFile -Path $PidFile | Out-Null
    }

    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        Start-Sleep -Milliseconds 250
        if (-not (Test-BackendHealthy -TargetUrl $HealthUrl)) {
            break
        }
    }

    if (Test-BackendHealthy -TargetUrl $HealthUrl) {
        Stop-BackendFromPidFile -Path $PidFile | Out-Null
        Start-Sleep -Milliseconds 500
    }

    Write-Host "Starting a fresh backend in this console."
}
elseif (Test-BackendHealthy -TargetUrl $HealthUrl) {
    Write-Host "Backend already running at $Url"
    if (-not $NoBrowser) {
        Start-Process $Url | Out-Null
    }
    return
}

$launch = Get-PythonLaunch -ScriptPath $BackendScript -BackendHost $BackendHost -BackendPort $Port -UseForceCpu $ForceCpu.IsPresent -UseAllowGpu $AllowGpu.IsPresent

Write-Host "Starting backend: $($launch.FilePath) $($launch.Arguments -join ' ')"

if ($Detached) {
    $process = Start-Process -FilePath $launch.FilePath `
        -ArgumentList $launch.Arguments `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -PassThru
    Set-Content -Path $PidFile -Value ("process:" + $process.Id) -Encoding ascii

    if (-not (Wait-BackendHealthy -TargetUrl $HealthUrl)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        if (Test-Path $PidFile) {
            Remove-Item -LiteralPath $PidFile -Force
        }
        throw "Backend did not become ready at $HealthUrl"
    }

    Write-Host "Backend is ready at $Url"
    if (-not $NoBrowser) {
        Start-Process $Url | Out-Null
    }
    Write-Host "Backend is running detached. Closing this console will not stop it."
    return
}

if (-not $NoBrowser) {
    $browserScript = {
        param([string]$TargetHealthUrl, [string]$TargetUrl)
        for ($attempt = 0; $attempt -lt 80; $attempt++) {
            Start-Sleep -Milliseconds 500
            try {
                $response = Invoke-WebRequest -Uri $TargetHealthUrl -UseBasicParsing -TimeoutSec 2
                if ($response.StatusCode -eq 200) {
                    Start-Process $TargetUrl | Out-Null
                    return
                }
            }
            catch {
            }
        }
    }
    Start-Job -ScriptBlock $browserScript -ArgumentList $HealthUrl, $Url | Out-Null
}
Write-Host "Backend will run in this console. Close this console or press Ctrl+C to stop it."
Push-Location $ProjectRoot
try {
    do {
        if (Test-Path $RestartFlagFile) {
            Remove-Item -LiteralPath $RestartFlagFile -Force -ErrorAction SilentlyContinue
        }
        $env:SNAKE_RESTART_IN_PARENT = "1"
        & $launch.FilePath @($launch.Arguments)
        $exitCode = $LASTEXITCODE
        Remove-Item Env:SNAKE_RESTART_IN_PARENT -ErrorAction SilentlyContinue
        $restartRequested = ($exitCode -eq 42) -or (Test-Path $RestartFlagFile)

        if ($restartRequested) {
            Remove-Item -LiteralPath $RestartFlagFile -Force -ErrorAction SilentlyContinue
            Write-Host "Backend restart requested. Starting again in the same console..."
            Start-Sleep -Milliseconds 500
        }
    } while ($restartRequested)

    if ($exitCode -ne 0) {
        Write-Host "Backend exited with code $exitCode"
    }
}
finally {
    Remove-Item Env:SNAKE_RESTART_IN_PARENT -ErrorAction SilentlyContinue
    Pop-Location
}

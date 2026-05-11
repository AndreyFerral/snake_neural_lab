Add-Type -AssemblyName System.Windows.Forms

$DefaultProjectPath = "C:\Users\Andrey\Documents\Codex\Snake\snake_neural_lab"
$LmStudioBase = "http://localhost:1234/v1"
$PythonVersion = "3.12"

Write-Host ""
Write-Host "=== Aider Web Picker ==="
Write-Host ""

# -----------------------------
# 1. Выбор папки проекта
# -----------------------------

Write-Host "Папка проекта по умолчанию:"
Write-Host $DefaultProjectPath
Write-Host ""
Write-Host "Enter  = использовать папку по умолчанию"
Write-Host "b      = выбрать папку через окно"
Write-Host "или вставь путь вручную"
Write-Host ""

$projectChoice = Read-Host "Папка проекта"

if ([string]::IsNullOrWhiteSpace($projectChoice)) {
    $ProjectPath = $DefaultProjectPath
}
elseif ($projectChoice -eq "b") {
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = "Выбери папку проекта для aider"
    $dialog.ShowNewFolderButton = $true

    if (Test-Path $DefaultProjectPath) {
        $dialog.SelectedPath = $DefaultProjectPath
    }

    $result = $dialog.ShowDialog()

    if ($result -ne [System.Windows.Forms.DialogResult]::OK) {
        Write-Host "Папка не выбрана. Выход."
        pause
        exit 1
    }

    $ProjectPath = $dialog.SelectedPath
}
else {
    $ProjectPath = $projectChoice.Trim('"')
}

if (-not (Test-Path $ProjectPath)) {
    Write-Host ""
    Write-Host "ОШИБКА: папка проекта не найдена:"
    Write-Host $ProjectPath
    pause
    exit 1
}

Set-Location $ProjectPath

# -----------------------------
# 2. Проверка git
# -----------------------------

try {
    git --version | Out-Null
} catch {
    Write-Host "ОШИБКА: git не найден. Установи Git for Windows."
    pause
    exit 1
}

if (-not (Test-Path ".git")) {
    Write-Host ""
    Write-Host "В папке нет git-репозитория."
    $initGit = Read-Host "Создать git repo здесь? y/n"

    if ($initGit -eq "y") {
        git init
    } else {
        Write-Host "aider лучше работает внутри git repo. Выход."
        pause
        exit 1
    }
}

# -----------------------------
# 3. Папка истории aider вне проекта
# -----------------------------

$SafeProjectName = Split-Path $ProjectPath -Leaf
$StateDir = ".aider-state"
New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

if (Test-Path ".aider.chat.history.md") {
    Move-Item ".aider.chat.history.md" "$StateDir\chat.history.md" -Force
}

if (Test-Path ".aider.input.history") {
    Move-Item ".aider.input.history" "$StateDir\input.history" -Force
}

if (Test-Path ".aider.llm.history") {
    Move-Item ".aider.llm.history" "$StateDir\llm.history" -Force
}

if (Test-Path ".web_backend.pid") {
    Remove-Item ".web_backend.pid" -Force
}

if (Test-Path "__pycache__") {
    Remove-Item "__pycache__" -Recurse -Force
}

# -----------------------------
# 4. LM Studio + выбор модели
# -----------------------------

$env:LM_STUDIO_API_KEY = "dummy-api-key"
$env:LM_STUDIO_API_BASE = $LmStudioBase

Write-Host ""
Write-Host "Проверяю LM Studio..."
Write-Host "Перед запуском должен быть включён:"
Write-Host "LM Studio -> Developer -> Start Server"
Write-Host ""

try {
    $modelsResponse = Invoke-RestMethod "$LmStudioBase/models"

    # ВАЖНО: сортировка моделей по алфавиту
    $models = @(
        $modelsResponse.data |
        ForEach-Object { $_.id } |
        Sort-Object
    )
} catch {
    Write-Host "ОШИБКА: не удалось подключиться к LM Studio."
    Write-Host "Проверь, что сервер запущен:"
    Write-Host $LmStudioBase
    pause
    exit 1
}

if (-not $models -or $models.Count -eq 0) {
    Write-Host "ОШИБКА: LM Studio не вернул список моделей."
    pause
    exit 1
}

Write-Host "Доступные модели LM Studio:"
Write-Host ""

for ($i = 0; $i -lt $models.Count; $i++) {
    Write-Host "$($i + 1). $($models[$i])"
}

Write-Host ""
$modelChoice = Read-Host "Выбери номер модели, Enter = 1"

if ([string]::IsNullOrWhiteSpace($modelChoice)) {
    $modelChoice = "1"
}

if (-not ($modelChoice -as [int])) {
    Write-Host "Неверный выбор модели."
    pause
    exit 1
}

$modelNumber = [int]$modelChoice

if ($modelNumber -lt 1 -or $modelNumber -gt $models.Count) {
    Write-Host "Неверный номер модели."
    pause
    exit 1
}

$SelectedModel = $models[$modelNumber - 1]
$AiderModel = "lm_studio/$SelectedModel"

# -----------------------------
# 5. Запуск aider web
# -----------------------------

Write-Host ""
Write-Host "Запускаю aider web..."
Write-Host ""
Write-Host "Проект:"
Write-Host $ProjectPath
Write-Host ""
Write-Host "Модель:"
Write-Host $AiderModel
Write-Host ""
Write-Host "История aider:"
Write-Host $StateDir
Write-Host ""

$AiderArgs = @(
    "--browser",
    "--model", $AiderModel,
    "--input-history-file", "$StateDir\input.history",
    "--chat-history-file", "$StateDir\chat.history.md",
	"--read", "README.md",   # <--- Файл всегда будет загружен как справочник
    "--no-show-model-warnings"
)

& py "-$PythonVersion" -m aider @AiderArgs

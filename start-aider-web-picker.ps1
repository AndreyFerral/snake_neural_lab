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
# 4. LM Studio + выбор моделей (Architect & Editor)
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
    $models = @(
        $modelsResponse.data |
        ForEach-Object { $_.id } |
        Sort-Object
    )
} catch {
    Write-Host "ОШИБКА: не удалось подключиться к LM Studio."
    Write-Host "Проверь, что сервер запущен: $LmStudioBase"
    pause
    exit 1
}

if (-not $models -or $models.Count -eq 0) {
    Write-Host "ОШИБКА: LM Studio не вернул список моделей."
    pause
    exit 1
}

Write-Host "Доступные модели в LM Studio:"
Write-Host ""
for ($i = 0; $i -lt $models.Count; $i++) {
    Write-Host "$($i + 1). $($models[$i])"
}
Write-Host ""

# Выбор Архитектора
$archChoice = Read-Host "Выбери модель для ARCHITECT (Мозг, например qwen3.6-35b-a3b)"
if ([string]::IsNullOrWhiteSpace($archChoice)) { $archChoice = "1" }
$archNum = [int]$archChoice
$ArchitectModel = "lm_studio/" + $models[$archNum - 1]

# Выбор Редактора
$editChoice = Read-Host "Выбери модель для EDITOR (Правки, например gemma-4-26b-a4b-it)"
if ([string]::IsNullOrWhiteSpace($editChoice)) { 
    $EditorModel = $ArchitectModel 
} else {
    $editNum = [int]$editChoice
    $EditorModel = "lm_studio/" + $models[$editNum - 1]
}

# -----------------------------
# 5. Запуск aider web
# -----------------------------

Write-Host ""
Write-Host "=== КОНФИГУРАЦИЯ ЗАПУСКА ==="
Write-Host "Проект:    $ProjectPath"
Write-Host "Architect: $ArchitectModel"
Write-Host "Editor:    $EditorModel"
Write-Host "============================"
Write-Host ""

$AiderArgs = @(
    "--browser",
    
    # Модели
    "--model", $ArchitectModel,
    "--architect",
    "--editor-model", $EditorModel,
    
    # Оптимизация для локалок и экономии контекста
    "--edit-format", "diff",        # Заставляет локальные модели присылать только изменения
    "--map-tokens", "1024",         # Ограничивает размер "карты проекта", чтобы не жрать контекст
    "--no-stream",                  # Локальные модели иногда стабильнее работают без стриминга в файлы
    
    # История и файлы
    "--input-history-file", "$StateDir\input.history",
    "--chat-history-file", "$StateDir\chat.history.md",
    "--no-auto-commit",
    "--no-show-model-warnings"
)

# Если вы хотите добавить README.md по умолчанию, раскомментируйте строку ниже
# $AiderArgs += "--read", "README.md"

& py "-$PythonVersion" -m aider @AiderArgs

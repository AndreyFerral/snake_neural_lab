Add-Type -AssemblyName System.Windows.Forms

# =========================================================
# AIDER + LM STUDIO WEB LAUNCHER
# Optimized for large-context local models (132k)
# =========================================================

$DefaultProjectPath = "C:\Users\Andrey\Documents\Codex\Snake\snake_neural_lab"

$LmStudioBase = "http://localhost:1234/v1"

$PythonVersion = "3.12"

# ---------------------------------------------------------
# LM Studio / Aider environment
# ---------------------------------------------------------

$env:LM_STUDIO_API_KEY = "dummy-api-key"
$env:LM_STUDIO_API_BASE = $LmStudioBase

# Увеличенный timeout для больших diff и длинных ответов
$env:AIDER_OPENAI_API_TIMEOUT = "600"

Write-Host ""
Write-Host "================================================="
Write-Host "              AIDER WEB PICKER"
Write-Host "================================================="
Write-Host ""

# =========================================================
# 1. Выбор папки проекта
# =========================================================

Write-Host "Папка проекта по умолчанию:"
Write-Host $DefaultProjectPath
Write-Host ""

Write-Host "Enter = использовать папку по умолчанию"
Write-Host "b     = выбрать папку через окно"
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

        Write-Host ""
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

# =========================================================
# 2. Проверка git
# =========================================================

try {

    git --version | Out-Null

}
catch {

    Write-Host ""
    Write-Host "ОШИБКА: git не найден."
    Write-Host "Установи Git for Windows."

    pause
    exit 1
}

if (-not (Test-Path ".git")) {

    Write-Host ""
    Write-Host "В папке нет git-репозитория."

    $initGit = Read-Host "Создать git repo здесь? y/n"

    if ($initGit -eq "y") {

        git init

    }
    else {

        Write-Host ""
        Write-Host "aider лучше работает внутри git repo."

        pause
        exit 1
    }
}

# =========================================================
# 3. Папка состояния aider
# =========================================================

$StateDir = ".aider-state"

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

# Перенос старых файлов истории
if (Test-Path ".aider.chat.history.md") {
    Move-Item ".aider.chat.history.md" "$StateDir\chat.history.md" -Force
}

if (Test-Path ".aider.input.history") {
    Move-Item ".aider.input.history" "$StateDir\input.history" -Force
}

if (Test-Path ".aider.llm.history") {
    Move-Item ".aider.llm.history" "$StateDir\llm.history" -Force
}

# Cleanup
$RuntimeDir = ".runtime"
# Убедимся, что папка существует (на случай, если её ещё нет)
if (-not (Test-Path $RuntimeDir)) {
    New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
}
# Удаляем PID из .runtime
if (Test-Path "$RuntimeDir/web_backend.pid") {
    Remove-Item "$RuntimeDir/web_backend.pid" -Force
}
# Очистка Python кэша
if (Test-Path "__pycache__") {
    Remove-Item "__pycache__" -Recurse -Force
}

# =========================================================
# 4. Создание .aiderignore если отсутствует
# =========================================================

$AiderIgnorePath = ".aiderignore"

if (-not (Test-Path $AiderIgnorePath)) {

@"
.aider-state/
venv/
.venv/
node_modules/
dist/
build/
coverage/
__pycache__/
*.log
*.tmp
*.cache
*.bak
*.pyc
"@ | Out-File -Encoding UTF8 $AiderIgnorePath

    Write-Host ""
    Write-Host ".aiderignore создан."
}

# =========================================================
# 5. Проверка LM Studio
# =========================================================

Write-Host ""
Write-Host "Проверяю LM Studio..."
Write-Host ""
Write-Host "LM Studio должен быть запущен:"
Write-Host "Developer -> Start Server"
Write-Host ""

try {

    $modelsResponse = Invoke-RestMethod "$LmStudioBase/models"

    $models = @(
        $modelsResponse.data |
        ForEach-Object { $_.id } |
        Sort-Object
    )

}
catch {

    Write-Host ""
    Write-Host "ОШИБКА: не удалось подключиться к LM Studio."
    Write-Host "Проверь сервер:"
    Write-Host $LmStudioBase

    pause
    exit 1
}

if (-not $models -or $models.Count -eq 0) {

    Write-Host ""
    Write-Host "ОШИБКА: LM Studio не вернул список моделей."

    pause
    exit 1
}

# =========================================================
# 6. Вывод моделей
# =========================================================

Write-Host ""
Write-Host "================================================="
Write-Host "           ДОСТУПНЫЕ МОДЕЛИ"
Write-Host "================================================="
Write-Host ""

for ($i = 0; $i -lt $models.Count; $i++) {

    Write-Host "$($i + 1). $($models[$i])"
}

Write-Host ""

# =========================================================
# 7. Рекомендации
# =========================================================

Write-Host "================================================="
Write-Host "РЕКОМЕНДУЕМЫЕ МОДЕЛИ"
Write-Host "================================================="
Write-Host ""
Write-Host "ЛУЧШИЙ ОБЩИЙ ВАРИАНТ: qwen3.6-35b-a3b"
Write-Host "ЛУЧШИЙ ДЛЯ КОДА: qwen3-coder-30b-a3b-instruct"
Write-Host "ЛУЧШИЙ БАЛАНС: qwen3.6-27b"
Write-Host "НЕ РЕКОМЕНДУЕТСЯ как основная aider-модель: gpt-oss-20b"
Write-Host ""

# =========================================================
# 8. Выбор режима работы
# =========================================================

Write-Host "================================================="
Write-Host "РЕЖИМ РАБОТЫ"
Write-Host "================================================="
Write-Host ""
Write-Host "1 = Single Model (РЕКОМЕНДУЕТСЯ)"
Write-Host "2 = Architect + Editor"
Write-Host ""

$ModeChoice = Read-Host "Выбор режима"

if ([string]::IsNullOrWhiteSpace($ModeChoice)) {
    $ModeChoice = "1"
}

# =========================================================
# 9. Выбор основной модели
# =========================================================

Write-Host ""
Write-Host "Выбери ОСНОВНУЮ модель:"
Write-Host ""

$modelChoice = Read-Host "Номер модели"

if ([string]::IsNullOrWhiteSpace($modelChoice)) {
    $modelChoice = "1"
}

$modelNum = [int]$modelChoice

if ($modelNum -lt 1 -or $modelNum -gt $models.Count) {

    Write-Host ""
    Write-Host "Неверный номер модели."

    pause
    exit 1
}

$MainModel = "lm_studio/" + $models[$modelNum - 1]

# =========================================================
# 10. Architect + Editor mode
# =========================================================

$UseArchitect = $false
$EditorModel = ""

if ($ModeChoice -eq "2") {

    $UseArchitect = $true

    Write-Host ""
    Write-Host "Выбери EDITOR модель:"
    Write-Host ""

    $editChoice = Read-Host "Номер editor модели"

    if ([string]::IsNullOrWhiteSpace($editChoice)) {

        $EditorModel = $MainModel
    }
    else {

        $editNum = [int]$editChoice

        if ($editNum -lt 1 -or $editNum -gt $models.Count) {

            Write-Host ""
            Write-Host "Неверный номер editor модели."

            pause
            exit 1
        }

        $EditorModel = "lm_studio/" + $models[$editNum - 1]
    }
}

# =========================================================
# 11. Конфигурация aider
# =========================================================

$AiderArgs = @(

    # Web UI
    "--browser",

    # Основная модель
    "--model", $MainModel,

    # Лучший формат для локальных LLM
    "--edit-format", "diff",

    # Больше map
    "--map-tokens", "2048",

    # Reuse prompts/context
    "--cache-prompts",

    # Ограничение по текущему subtree
    "--subtree-only",

    # Стабильнее для локальных моделей
    "--no-stream",

    # История
    "--input-history-file", "$StateDir\input.history",

    "--chat-history-file", "$StateDir\chat.history.md",

    # Git
    "--no-auto-commit",

    # Меньше мусора
    "--no-show-model-warnings"
)

# Architect mode
if ($UseArchitect) {

    $AiderArgs += "--architect"

    $AiderArgs += "--editor-model"
    $AiderArgs += $EditorModel
}

# =========================================================
# 12. Итоговая конфигурация
# =========================================================

Write-Host ""
Write-Host "================================================="
Write-Host "          КОНФИГУРАЦИЯ ЗАПУСКА"
Write-Host "================================================="
Write-Host ""

Write-Host "Проект:"
Write-Host "  $ProjectPath"
Write-Host ""

Write-Host "Основная модель:"
Write-Host "  $MainModel"
Write-Host ""

if ($UseArchitect) {

    Write-Host "Editor модель:"
    Write-Host "  $EditorModel"
    Write-Host ""
}

Write-Host "Context strategy:"
Write-Host "  map-tokens = 2048"
Write-Host ""

Write-Host "================================================="
Write-Host ""

# =========================================================
# 13. Запуск aider
# =========================================================

& py "-$PythonVersion" -m aider @AiderArgs
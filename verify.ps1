$ErrorActionPreference = "Stop"
$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Error "Не найдено виртуальное окружение. Создайте его и установите зависимости по инструкции в README.md."
    exit 1
}

Push-Location -LiteralPath $PSScriptRoot
try {
    & $pythonExe -m unittest discover -s tests -v
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

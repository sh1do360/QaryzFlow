$ErrorActionPreference = "Stop"
$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Error "Не найдено виртуальное окружение. Создайте его и установите зависимости по инструкции в README.md."
    exit 1
}

Push-Location -LiteralPath $PSScriptRoot
try {
    & $pythonExe -m streamlit run app.py --server.headless false --server.address 127.0.0.1 --server.port 8501
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

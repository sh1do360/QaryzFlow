param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Не найдено виртуальное окружение. Выполните py -m venv .venv, затем .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

Push-Location -LiteralPath $PSScriptRoot
try {
    & $pythonExe -c "import streamlit, sklearn"
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось загрузить зависимости. Выполните .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    }

    Write-Host "QaryzFlow открывается по адресу http://127.0.0.1:$Port"
    & $pythonExe -m streamlit run app.py --server.headless false --server.address 127.0.0.1 --server.port $Port
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

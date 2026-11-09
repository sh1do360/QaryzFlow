$ErrorActionPreference = "Stop"
$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Не найдено виртуальное окружение. Выполните py -m venv .venv, затем .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

Push-Location -LiteralPath $PSScriptRoot
try {
    & $pythonExe -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    & $pythonExe -m scripts.check_reference
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

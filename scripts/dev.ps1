#!/usr/bin/env pwsh
# Dev task runner. Usage: .\scripts\dev.ps1 <task>
# Tasks: install | test | app | dataset | smoke

param([string]$Task = "help")

$ErrorActionPreference = "Stop"

switch ($Task) {
    "install" {
        pip install -r requirements.txt
    }
    "test" {
        pytest -v
    }
    "app" {
        python app.py
    }
    "dataset" {
        python dataset/prepare_dataset.py
    }
    "smoke" {
        pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw "install failed" }
        pytest -v
        if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
        python -c "import numpy, scipy, PIL, matplotlib, gradio, insightface, onnxruntime; print('imports OK')"
    }
    default {
        Write-Host "Usage: .\scripts\dev.ps1 <install|test|app|dataset|smoke>"
        Write-Host ""
        Write-Host "  install   pip install -r requirements.txt"
        Write-Host "  test      pytest -v"
        Write-Host "  app       python app.py (launch Gradio)"
        Write-Host "  dataset   download LFW dataset images"
        Write-Host "  smoke     install + test + import sanity"
    }
}

# setup.ps1
# Automatiseert de omgevingssetup op een nieuwe machine.
# Uitvoeren vanuit de hoofdmap van het project: .\setup.ps1

Write-Host "Zomboid Survival Assistant -- setup starten..." -ForegroundColor Cyan

# Virtuele omgeving aanmaken indien nog niet aanwezig
if (-Not (Test-Path "venv")) {
    Write-Host "Virtuele omgeving aanmaken..."
    python -m venv venv
} else {
    Write-Host "Virtuele omgeving bestaat al, wordt overgeslagen."
}

# Activeren
Write-Host "Virtuele omgeving activeren..."
& venv\Scripts\Activate.ps1

# Dependencies installeren
Write-Host "Dependencies installeren vanuit requirements.txt..."
pip install -r requirements.txt --quiet

# CUDA-check
Write-Host "`nControle GPU/CUDA-status..." -ForegroundColor Cyan
$cudaCheck = python -c "import torch; print(torch.cuda.is_available())"
if ($cudaCheck -eq "True") {
    Write-Host "CUDA werkt correct, GPU wordt gebruikt." -ForegroundColor Green
} else {
    Write-Host "CUDA niet beschikbaar -- CPU-only PyTorch geinstalleerd, of geen NVIDIA-GPU aanwezig." -ForegroundColor Yellow
    Write-Host "Indien je wel een NVIDIA-GPU hebt, voer uit:" -ForegroundColor Yellow
    Write-Host "  pip uninstall torch torchvision -y" -ForegroundColor Yellow
    Write-Host "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130" -ForegroundColor Yellow
}

# Modelbestand check
Write-Host "`nControle modelbestand..." -ForegroundColor Cyan
if (Test-Path "models\zomboid_v1.pt") {
    Write-Host "Model gevonden op models\zomboid_v1.pt" -ForegroundColor Green
} else {
    Write-Host "Model NIET gevonden." -ForegroundColor Red
}

# .env check
Write-Host "`nControle .env-bestand..." -ForegroundColor Cyan
if (Test-Path ".env") {
    Write-Host ".env gevonden." -ForegroundColor Green
} else {
    Write-Host ".env niet gevonden (optioneel, enkel nodig voor GenAI-advies)." -ForegroundColor Yellow
    Write-Host "Maak indien gewenst een .env-bestand aan met: GEMINI_API_KEY=jouw_key" -ForegroundColor Yellow
}

Write-Host "`nSetup klaar. Test met: python src\risk_score.py data\dataset\test\images" -ForegroundColor Cyan

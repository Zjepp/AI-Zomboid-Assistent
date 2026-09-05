# Project Zomboid AI Survival Assistant

A computer vision + GenAI pipeline that analyzes Project Zomboid screenshots, detects zombies, fire, doors, windows, trees, and the player, computes a real-time risk score, suggests the safest compass direction to flee, and can optionally generate natural-language warnings.

## Features

- Custom-trained YOLOv8n object detector (7 classes: `Door`, `Fire`, `Player`, `Tree`, `Zombie`, `Zombie Dead`, `window`)
- Rule-based risk-scoring engine using player-relative distance
- Isometric-camera-aware compass direction system (suggests safest direction(s) to flee)
- Local, unlimited rule-based warning generator, with optional rate-limited GenAI (Gemini) advisor for richer commentary
- Live background assistant: captures your screen while you play and gives real-time warnings

## Requirements

- **Python 3.11–3.13**
- **Windows** (scripts and setup below are Windows/PowerShell-specific; core Python code is cross-platform)
- **NVIDIA GPU** strongly recommended for training (CPU works but is much slower); not required for running the live assistant with an already-trained model
- **Google AI Studio API key** (free tier) — only needed if you want GenAI-generated advice; the app runs fully without it
- See `requirements.txt` for exact Python package versions

## Repository Structure

```
zomboid-survival-assistant/
├── data/
│   ├── raw/              # collected screenshots (not in git — see below)
│   ├── annotated/        # Roboflow exports / pre-labels (not in git)
│   └── dataset/          # final train/val/test split (not in git)
├── models/
│   └── zomboid_v1.pt     # trained model weights (included in git — ~6MB, small enough to commit)
├── runs/                 # training run outputs, logs, plots (not in git)
├── src/                  # all project code (see script overview below)
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone and enter the repo

```powershell
git clone https://github.com/Zjepp/AI-Zomboid-Assistent
cd zomboid-survival-assistant
```

### 2. Create and activate a virtual environment

Requires Python 3.11, 3.12, or 3.13.

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. GenAI setup (optional, not required to use the app)

Only needed if you want AI-generated advice instead of/alongside the built-in local rule-based one. Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

## Quick Start — Using the Trained Model

Once you've completed steps 1–2 above (repo cloned, venv set up, dependencies installed), you're ready to go — the trained model (`models/zomboid_v1.pt`) is already included in the repo. **None of the steps below require any dataset, annotation, or training.**

### Run the live assistant (main feature)

Starts capturing your screen every few seconds while you play, and prints risk warnings to the console.

```powershell
python src\live_assistant.py
```

Press `Ctrl+C` to stop. Run this with Project Zomboid in windowed or borderless mode (not exclusive fullscreen).

### Analyze a single screenshot or folder

```powershell
python src\risk_score.py path\to\screenshot.jpg
python src\risk_score.py path\to\folder_of_images
```

### Run the full pipeline (detection + risk score + GenAI advice) on one image

```powershell
python src\pipeline.py path\to\screenshot.jpg
```

### Get a GenAI-generated warning for one screenshot (optional, requires `.env` setup)

```powershell
python src\genai_advisor.py
```

(Edit the image path at the bottom of the script to change which image is analyzed. Uses your daily API quota — use sparingly.)

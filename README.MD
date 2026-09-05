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
│   └── zomboid_v1.pt     # trained model weights (not in git — see Setup)
├── runs/                 # training run outputs, logs, plots (not in git)
├── src/                  # all project code (see script overview below)
├── requirements.txt
├── setup.ps1
└── README.md
```

## Setup

### 1. Clone and enter the repo

```powershell
git clone https://github.com/<your-username>/zomboid-survival-assistant.git
cd zomboid-survival-assistant
```

### 2. Run the setup script

```powershell
.\setup.ps1
```

This creates a virtual environment, installs all dependencies, and checks your GPU/CUDA, model file, and `.env` status. If you'd rather do it manually:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. GPU support (optional but recommended)

If you have an NVIDIA GPU, reinstall PyTorch with CUDA support (check your CUDA version first with `nvidia-smi`):

```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

The app also runs fine on CPU-only PyTorch, just slower per detection (a fraction of a second either way for a single image, so this mostly matters for training, not for using the app).

### 4. Get the trained model — the only thing you need to actually use the app

Download `zomboid_v1.pt` from the [Releases page](#) of this repo and place it at:

```
models/zomboid_v1.pt
```

Model weights aren't stored in git (see "What's Not in Git" below) — once this file is in place, you can run everything in the **Quick Start** section without needing any dataset, annotation tool, or training step.

### 5. GenAI setup (optional, not required to use the app)

Only needed if you want AI-generated advice instead of/alongside the built-in local rule-based one. Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

Get a free key at https://aistudio.google.com/apikey. Note: the free tier has a low daily request limit, which is why the live assistant defaults to a local, unlimited rule-based advisor instead — GenAI is an optional extra, never required.

## Quick Start — Using the Trained Model

Once you've completed steps 1–4 above (venv set up, dependencies installed, `zomboid_v1.pt` in `models/`), you're ready to go. **None of the steps below require any dataset, annotation, or training.**

### Run the live assistant (main feature)

Starts capturing your screen every few seconds while you play, and prints risk warnings to the console.

```powershell
python src\live_assistant.py
```

Press `Ctrl+C` to stop. Run this with Project Zomboid in windowed or borderless mode (not exclusive fullscreen).

**Before first use:** verify the compass direction calibration matches your setup — check `ISO_OFFSET_DEGREES` in `risk_score.py` against your in-game compass (see comments in that file).

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

## Advanced — Training Your Own Model

Everything below this point is only needed if you want to **retrain or extend the model yourself**, not to use the already-trained one.

### Requirements for training

- A Roboflow account (free tier) for annotation
- Your own collected screenshots or gameplay recordings
- Significantly more time and, ideally, a GPU

### Data collection / preparation scripts

- `extract_frames.py` — extracts diverse, de-duplicated frames from recorded gameplay video
- `organise_screenshots.py` — renames/organizes raw screenshots sequentially
- `convert_jfif_to_jpg.py` — converts any `.jfif` images to `.jpg`

### Train the model

Requires an annotated dataset at `data/dataset/` (exported from Roboflow in YOLO format):

```powershell
python src\train.py
```

### Evaluate the trained model

```powershell
python src\evaluate.py
```

### Visualize predictions on test images

```powershell
python src\visualize_pred.py
```

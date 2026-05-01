# Velocity3D

AI-powered 3D model generation desktop app built with Electron + React + Three.js + Python (bpy 5.1.1).

## Requirements

- Node.js 20+
- Python 3.13
- CUDA-capable GPU (recommended, 6GB+ VRAM)

## Development Setup

### 1. Install Node.js dependencies
```bash
npm install
```

### 2. Set up Python environment
```bash
cd backend
python3.13 -m venv venv
venv/Scripts/activate  # Windows
# or: source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
# Install Shap-E (text-to-3D, has setup.py):
pip install git+https://github.com/openai/shap-e.git
# Install PyTorch with CUDA (pick your version):
# CUDA 12.1: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU only:  pip install torch torchvision
```

### 3. Run in development mode
```bash
npm run dev
```

## Running Tests

### Python backend tests
```bash
cd backend
pytest tests/ -v
```

### Frontend tests
```bash
npm test
```

## Building for Production

```bash
npm run package
```

Installers will be output to `release/`.

## Architecture

- **Frontend**: Electron + React + Three.js (WebGL viewport)
- **Backend**: Python 3.13 + FastAPI + bpy 5.1.1
- **Text-to-3D**: Shap-E
- **Image-to-3D**: TripoSR
- **IPC**: Local HTTP (127.0.0.1 only) via contextBridge

## Generated Models

Models are saved to `~/.velocity3d/models/` and history is stored in `~/.velocity3d/history.json` (Electron userData directory).

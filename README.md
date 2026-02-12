# Methane Detection with Deep Learning

A video course on simulated spectroscopy and 1D-CNN inference using PyTorch + ROCm.

## Course Overview

Build a complete ML pipeline — from physics-based simulation to trained inference model — for predicting methane mole fraction from simulated laser absorption scans. Development is split between your local machine (simulation, dataset generation, inference) and an AMD GPU cloud droplet (model training). This hybrid approach minimizes cloud costs while giving you full GPU acceleration where it matters.

**Prerequisites:** Basic Python, some familiarity with NumPy/Matplotlib, introductory understanding of machine learning concepts.

**Hardware/Platform:** Any Mac or Linux machine for local work; AMD Instinct GPU droplet (e.g. MI210/MI300X via cloud provider) for training only.

## Hybrid Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                      YOUR LOCAL MACHINE                      │
│                                                              │
│  Module 1A: Local Setup (pip install, directories)           │
│       ↓                                                      │
│  Module 2:  Physics Background (markdown)                    │
│       ↓                                                      │
│  Module 3:  HITRAN Simulation (numpy/hapi)                   │
│       ↓                                                      │
│  Module 4:  Detector Voltage Trace (numpy)                   │
│       ↓                                                      │
│  Module 5:  Dataset Generation (multiprocessing, h5py)       │
│       ↓                                                      │
│  dataset_1M.h5 ──── scp/rsync ────┐                         │
│                                    ↓                         │
│              ┌─────────────────────────────────────────┐     │
│              │          AMD GPU DROPLET                 │     │
│              │                                         │     │
│              │  Module 1B: GPU Session Setup            │     │
│              │       ↓                                 │     │
│              │  Module 6:  Build 1D-CNN (GPU)          │     │
│              │       ↓                                 │     │
│              │  Module 7:  Train Model (GPU) ──┐       │     │
│              │                                 │       │     │
│              └─────────────────────────────────┼───────┘     │
│                                                │             │
│  best_model.pt ←── scp/rsync ─────────────────┘             │
│       ↓                                                      │
│  Module 8:  Inference (CPU — or GPU if droplet is up)        │
│       ↓                                                      │
│  Module 9:  Packaging & Export                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Modules

### Local Machine (Modules 1A, 2–5, 8–9)

| Module | Notebook | Description |
|--------|----------|-------------|
| 1A | `Module_1A_Local_Setup.ipynb` | Install packages (PyTorch CPU, HAPI, NumPy), create directories, fetch HITRAN data |
| 2 | `Module_02_Physics_Background.ipynb` | Beer-Lambert Law, TDLAS, the forward/inverse problem |
| 3 | `Module_03_HITRAN_Simulation.ipynb` | Generate CH₄ absorbance spectra with HITRAN/HAPI |
| 4 | `Module_04_Detector_Voltage_Trace.ipynb` | Simulate laser scanning and detector output |
| 5 | `Module_05_Dataset_Generation.ipynb` | Generate 1M training scans with multiprocessing |
| 8 | `Module_08_Inference.ipynb` | Run predictions on unknown scans, error analysis |
| 9 | `Module_09_Packaging.ipynb` | Export to ONNX, deployment considerations |

### AMD GPU Droplet (Modules 1B, 6–7)

| Module | Notebook | Description |
|--------|----------|-------------|
| 1B | `Module_1B_GPU_Session_Setup.ipynb` | Install PyTorch + ROCm, verify GPU, check dataset upload |
| 6 | `Module_06_Build_1D_CNN.ipynb` | Define model architecture, DataLoader, verify GPU forward pass |
| 7 | `Module_07_Training.ipynb` | Train with Adam/MSE, monitor convergence, checkpoint |

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/methane-ml-course.git
cd methane-ml-course
```

### 2. Run Module 1A (local setup)

```bash
jupyter lab
# Open Module_1A_Local_Setup.ipynb and run all cells (~2-3 min)
```

This installs PyTorch (CPU-only, ~200 MB), HAPI, and the science stack. It also downloads CH₄ spectroscopic data from HITRAN and creates the project directory structure under `~/methane-ml-course/`.

### 3. Work through Modules 2–5 locally

All spectroscopic simulation, physics exploration, and dataset generation run on your CPU. No GPU needed.

### 4. Upload dataset and switch to GPU droplet

```bash
# Upload the generated dataset
scp ~/methane-ml-course/data/datasets/dataset_1M.h5 \
    root@<DROPLET_IP>:/root/methane-ml-course/data/datasets/

# On the droplet: run Module 1B, then Modules 6-7
```

### 5. Download trained model and finish locally

```bash
# Download model back to your Mac
scp root@<DROPLET_IP>:/root/methane-ml-course/models/best_model.pt \
    ~/methane-ml-course/models/

# Run Modules 8-9 locally
```

## Cost Comparison

| Approach | GPU Hours | Estimated Cost |
|----------|-----------|---------------|
| Everything on GPU droplet | 10–20 hrs | $20–40+ |
| Hybrid (GPU for training only) | 2–4 hrs | $4–8 |

Snapshots stored between sessions cost ~$0.06/GB/month ($4–6/month typical).

## Project Structure

```
methane-ml-course/
├── notebooks/
│   ├── Module_1A_Local_Setup.ipynb
│   ├── Module_1B_GPU_Session_Setup.ipynb
│   ├── Module_02_Physics_Background.ipynb
│   ├── Module_03_HITRAN_Simulation.ipynb
│   ├── Module_04_Detector_Voltage_Trace.ipynb
│   ├── Module_05_Dataset_Generation.ipynb
│   ├── Module_06_Build_1D_CNN.ipynb
│   ├── Module_07_Training.ipynb
│   ├── Module_08_Inference.ipynb
│   └── Module_09_Packaging.ipynb
├── data/
│   ├── hitran/          ← HITRAN line data (auto-downloaded)
│   ├── spectra/         ← Individual saved spectra
│   └── datasets/        ← Training dataset (dataset_1M.h5)
├── models/              ← Trained model checkpoints
├── outputs/             ← Exported models (ONNX, etc.)
├── course_outline_V2_Hybrid_Compute.md
└── README.md
```

## Key Technologies

- **HITRAN / HAPI** — Spectroscopic database and Python API for molecular absorption data
- **PyTorch** — Model definition, training, and inference (CPU locally, ROCm on AMD GPU)
- **ROCm** — AMD's GPU compute platform (used on cloud droplet only)
- **HDF5 / h5py** — Efficient storage for the 1M-scan training dataset
- **NumPy / Matplotlib** — Simulation, computation, and visualization

## License

[Add your license here]

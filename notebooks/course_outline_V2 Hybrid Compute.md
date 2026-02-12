# Restructuring Plan: Local Mac + Remote GPU Hybrid Workflow

## The Core Insight

Looking at your 9-module pipeline, **only training and GPU verification actually need the AMD GPU**. Everything else — HITRAN simulation, physics, dataset generation, even inference — is CPU-bound NumPy/HAPI work that runs perfectly on a MacBook Pro.

## Module-by-Module Breakdown

| Module | Original | Hybrid | Why |
|--------|----------|--------|-----|
| 1. Session Setup | GPU droplet | **Split into 1A (local) + 1B (GPU)** | Local needs pip packages; GPU needs ROCm |
| 2. Physics Background | GPU droplet | **Local Mac** | Markdown only, no compute |
| 3. HITRAN Simulation | GPU droplet | **Local Mac** | Pure NumPy/HAPI, CPU-only |
| 4. Detector Voltage Trace | GPU droplet | **Local Mac** | Pure NumPy, CPU-only |
| 5. Dataset Generation | GPU droplet | **Local Mac** | CPU-bound multiprocessing — Mac is fine |
| 6. Building the 1D-CNN | GPU droplet | **GPU droplet** | GPU forward-pass verification |
| 7. Training | GPU droplet | **GPU droplet** | The main GPU workload |
| 8. Inference | GPU droplet | **Mostly local Mac** | CPU inference is fast for single scans |
| 9. Packaging & Next Steps | GPU droplet | **Local Mac** | ONNX export, no GPU needed |

## Estimated GPU Time Saved

- **Before:** GPU droplet running for all 9 modules (~10-20 hours of work sessions × $1.99/hr = $20-40+)
- **After:** GPU droplet for Modules 6-7 only (~2-4 hours × $1.99/hr = $4-8)
- **Savings:** ~80% reduction in GPU cloud costs

## What Changes

### Module 1A: Local Environment Setup (NEW — replaces Module 1 for local work)

Run once on your Mac. Installs:
- `hitran-api` (HAPI)
- `numpy`, `matplotlib`, `seaborn`
- `h5py`, `tqdm`, `pyyaml`
- `torch` (CPU-only — for local inference and model architecture prototyping)

**Does NOT install:** ROCm, torchvision with ROCm, ultralytics, albumentations, opencv

Creates the same directory structure under a local project folder.

### Module 1B: GPU Session Setup (MINIMAL — only for training sessions)

Streamlined version that:
- Installs PyTorch + ROCm
- Verifies GPU access
- Expects the dataset (`dataset_1M.h5`) to already be uploaded

**Does NOT need:** HITRAN data, HAPI, simulation packages (those ran locally)

### Modules 2-5: No Structural Changes Needed

These modules work as-is with two minor adjustments:
1. Remove hardcoded paths like `/root/methane-ml-course/` — use `Path.cwd()` or `Path.home() / 'methane-ml-course'`
2. Remove any `torch.cuda` references (there are none in Modules 2-5 currently)

### Module 5 → Module 6 Bridge: Dataset Upload

After generating `dataset_1M.h5` locally, students upload it to the GPU droplet:

```bash
# From your Mac
scp ./data/datasets/dataset_1M.h5 root@<DROPLET_IP>:/root/methane-ml-course/data/datasets/
```

Or use `rsync` for resumable transfers:
```bash
rsync -avP ./data/datasets/dataset_1M.h5 root@<DROPLET_IP>:/root/methane-ml-course/data/datasets/
```

### Module 8: Dual-Mode Inference

Add a device selection cell at the top:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running inference on: {device}")
```

Single-scan and test-set inference run fine on CPU. Batch inference of 1M scans benefits from GPU but isn't required.

### Module 9: Fully Local

Model export (torch.save, ONNX) is CPU-only.

## Revised Course Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    YOUR MACBOOK PRO                       │
│                                                          │
│  Module 1A: Local Setup (pip install, directories)       │
│       ↓                                                  │
│  Module 2: Physics Background (markdown)                 │
│       ↓                                                  │
│  Module 3: HITRAN Simulation (numpy/hapi)                │
│       ↓                                                  │
│  Module 4: Detector Voltage Trace (numpy)                │
│       ↓                                                  │
│  Module 5: Dataset Generation (multiprocessing, h5py)    │
│       ↓                                                  │
│  dataset_1M.h5 ──── scp/rsync ────┐                     │
│                                    ↓                     │
│              ┌─────────────────────────────────────┐     │
│              │        AMD GPU DROPLET              │     │
│              │                                     │     │
│              │  Module 1B: GPU Session Setup        │     │
│              │       ↓                             │     │
│              │  Module 6: Build 1D-CNN (GPU)       │     │
│              │       ↓                             │     │
│              │  Module 7: Train Model (GPU) ──┐    │     │
│              │                                │    │     │
│              └────────────────────────────────┼────┘     │
│                                               │          │
│  best_model.pt ←── scp/rsync ────────────────┘          │
│       ↓                                                  │
│  Module 8: Inference (CPU — or GPU if droplet is up)     │
│       ↓                                                  │
│  Module 9: Packaging & Export                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Key Path Changes

The hardcoded `/root/methane-ml-course/` paths in Module 03 need to become portable:

```python
# OLD (GPU-only, assumes root user in Docker)
HITRAN_DATA_DIR = Path('/root/methane-ml-course/data/hitran')

# NEW (works on both Mac and GPU droplet)
PROJECT_DIR = Path.home() / 'methane-ml-course'
HITRAN_DATA_DIR = PROJECT_DIR / 'data' / 'hitran'
```

## Dataset Generation Performance (Mac vs. GPU Droplet)

Module 5's dataset generation is CPU-bound (HITRAN Voigt calculations are NumPy). A modern MacBook Pro with an M-series chip will actually be *competitive* with the cloud droplet's CPU for this workload, since:

- Apple Silicon has strong single-core and multi-core NumPy performance
- No network latency or Docker overhead
- Native memory access (no container memory limits)

For 1M scans with `multiprocessing.Pool`, expect:
- M1/M2 Pro (8-10 cores): ~2-4 hours
- M3 Pro/Max (12+ cores): ~1-3 hours
- Cloud CPU (depends on instance): ~2-6 hours

## Course Outline Edits

Update the course overview paragraph from:

> "All development takes place on an AMD GPU cloud droplet using PyTorch with ROCm and Jupyter notebooks."

To:

> "Development is split between your local machine (simulation, dataset generation, inference) and an AMD GPU cloud droplet (model training). This hybrid approach minimizes cloud costs while giving you full GPU acceleration where it matters."

Update Module 1 description to cover both 1A and 1B variants.

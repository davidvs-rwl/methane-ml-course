# Methane Detection with Deep Learning

A hands-on video course on simulated spectroscopy and 1D-CNN inference using PyTorch with ROCm on AMD GPUs.

## Overview

Build a complete ML pipeline — from physics-based simulation to trained inference model — for predicting methane (CH₄) mole fraction from simulated laser absorption scans. All development runs on AMD Instinct MI300X GPUs via DigitalOcean GPU Droplets.

**Full course outline:** [`course_outline.md`](course_outline.md)

## Modules

| Module | Topic | Notebook | Status |
|--------|-------|----------|--------|
| 1 | Environment Setup & ROCm Fundamentals | [`Module_01_Session_Setup.ipynb`](notebooks/Module_01_Session_Setup.ipynb) | ✅ |
| 2 | Physics Background — Beer-Lambert Law | — | 🔜 |
| 3 | Simulating Absorbance Spectra with HITRAN | [`Module_03_HITRAN_Simulation.ipynb`](notebooks/Module_03_HITRAN_Simulation.ipynb) | ✅ |
| 4 | Simulating the Detector Voltage Trace | — | 🔜 |
| 5 | Generating a 1M-Scan Training Dataset | — | 🔜 |
| 6 | Building the 1D-CNN in PyTorch | — | 🔜 |
| 7 | Training the Model | — | 🔜 |
| 8 | Inference & Evaluation | — | 🔜 |
| 9 | Packaging & Next Steps | — | 🔜 |

## Quick Start

1. Create an AMD GPU Droplet on DigitalOcean (MI300X, ATL1 datacenter)
2. SSH in and set up the environment:
   ```bash
   apt update && apt install python3.12-venv -y
   cd /root/methane-ml-course
   python3 -m venv venv
   source venv/bin/activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
   pip install jupyterlab ipykernel hitran-api numpy matplotlib seaborn h5py tqdm pyyaml
   python -m ipykernel install --user --name=methane-ml --display-name="Methane ML (ROCm)"
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
   ```
3. Open `Module_01_Session_Setup.ipynb` and run all cells
4. Proceed through modules in order

## Infrastructure

- **GPU:** AMD Instinct MI300X (192 GB HBM3)
- **Platform:** DigitalOcean Gradient GPU Droplets
- **Framework:** PyTorch with ROCm 6.2+
- **Spectroscopy:** HITRAN database via HAPI
- **Environment:** Jupyter Lab + Python venv

## License

MIT

## Author

David — [Redwood Labs](https://github.com/davidvs-rwl)

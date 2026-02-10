# Methane Detection with Deep Learning
## A Video Course on Simulated Spectroscopy & 1D-CNN Inference Using PyTorch + ROCm

---

## Course Overview

Build a complete ML pipeline — from physics-based simulation to trained inference model — for predicting methane mole fraction from simulated laser absorption scans. All development takes place on an AMD GPU cloud droplet using PyTorch with ROCm and Jupyter notebooks.

**Prerequisites:** Basic Python, some familiarity with NumPy/Matplotlib, introductory understanding of machine learning concepts.

**Hardware/Platform:** AMD Instinct GPU droplet (e.g. MI210/MI300X via cloud provider), ROCm stack, Jupyter Lab.

---

## Module 1: Environment Setup & ROCm Fundamentals

### 1.1 — Provisioning Your AMD GPU Droplet
- Selecting an AMD GPU cloud instance (provider options, instance sizing)
- SSH access and initial system configuration
- Verifying the GPU is visible: `rocm-smi`, `rocminfo`

### 1.2 — Understanding the Docker Environment

The AMD Developer Cloud runs Jupyter inside a Docker container. This has important implications:

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                     HOST SYSTEM (Ubuntu)                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              DOCKER CONTAINER "rocm"                    │ │
│  │   /home/rocm-user/jupyter/  ← Jupyter working dir       │ │
│  │   • PyTorch + ROCm                                      │ │
│  │   • JupyterLab                                          │ │
│  │   • Your notebooks & data                               │ │
│  └─────────────────────────────────────────────────────────┘ │
│                         ↕ GPU passthrough                    │
│                    [ AMD MI300X GPU ]                        │
└─────────────────────────────────────────────────────────────┘
```

**Key Docker Commands (from SSH):**
- `docker exec -it rocm /bin/bash` — Enter the container shell
- `docker logs rocm` — View container logs
- `docker ps` — List running containers

### 1.3 — Session Setup Workflow

**Critical:** The Docker container does not persist installed packages between sessions. Every time you create a new droplet (even from a snapshot), you must run the Session Setup notebook first.

**What persists in snapshots:**
- ✅ Notebook files (`.ipynb`) in Jupyter's working directory
- ✅ Data files you create in `./data/`
- ✅ Host system configuration

**What does NOT persist:**
- ❌ Python packages installed via `pip install`
- ❌ HITRAN database cache
- ❌ Kernel state and variables

**Session Setup Notebook:** `Module_01_Session_Setup.ipynb`

This notebook installs all required packages, sets up HITRAN, and verifies the environment. **Run it first every time you start a new session.**

### 1.4 — Snapshot, Destroy & Restore Workflow

Managing cloud costs is critical — GPU instances cost ~$1.99/hour even when powered off.

#### Creating a Snapshot
1. Save all notebooks in Jupyter (Ctrl+S)
2. From SSH: `sudo poweroff` (recommended for data integrity)
3. In cloud console: **Backups & Snapshots** → **Snapshots** → **Take Snapshot**
4. Name it descriptively (e.g., `methane-course-module3-complete`)
5. Wait for completion (typically 5-15 minutes)

#### Destroying the Droplet
1. Verify snapshot shows "available"
2. Go to **Settings** → **Destroy**
3. Confirm destruction — **billing stops immediately**
4. Important: Powering off does NOT stop billing; only destroying does

#### Restoring from Snapshot
1. Go to **Images** → **Snapshots**
2. Click **More** → **Create Droplet** on your snapshot
3. Select the same GPU type (MI300X) and an available region
4. Launch — wait for droplet to be ready
5. Access Jupyter via the IP address shown
6. **Run `Module_01_Session_Setup.ipynb` first!**

#### Cost Comparison
| State | Cost |
|-------|------|
| Droplet running | ~$1.99/hour (~$48/day) |
| Droplet powered off | ~$1.99/hour (still charged!) |
| Snapshot stored | ~$0.06/GB/month (~$4-6/month) |
| Droplet destroyed | $0 |

**Deliverable:** A running Jupyter environment with PyTorch + ROCm + HITRAN ready, and understanding of the session/snapshot workflow.

---

## Module 2: Physics Background — Beer-Lambert Law & Laser Absorption Spectroscopy

### 2.1 — Beer-Lambert Law Primer
- The relationship: α = −ln(Iₜ / I₀), where α is absorbance
- What determines absorbance: absorption coefficient, path length (L), and mole fraction (ξ)
- Why absorbance scales linearly with concentration (at moderate optical depths)

### 2.2 — Tunable Diode Laser Absorption Spectroscopy (TDLAS)
- How a tunable diode laser scans across a wavelength range
- The incident intensity baseline I₀ as a function of drive voltage
- Wavenumber (cm⁻¹) vs. wavelength (nm): the conversion nm = 10⁷ / ν
- Why we chose the 4383–4386 cm⁻¹ region for CH₄ (three transitions spanning ~1 order of magnitude in peak absorbance, minimal interference from H₂O, CO₂, CO)

### 2.3 — From Physics to Data: What the CNN Will Learn
- The forward problem: (T, P, ξ) → absorbance spectrum → detector voltage trace
- The inverse problem the CNN solves: detector voltage trace + (T, P) → ξ
- Why ML is attractive here: eliminating manual baseline fitting, handling high-absorbance regimes

**Deliverable:** Conceptual understanding. No code yet — just notebook markdown cells with equations and diagrams.

---

## Module 3: Simulating Absorbance Spectra with HITRAN

### 3.1 — Introduction to HITRAN and HAPI
- What HITRAN is: a spectroscopic database of molecular transition parameters
- The HAPI Python library: `fetch()`, `absorptionCoefficient_Voigt()`
- Note: HITRAN data is fetched during Session Setup and cached in `./data/hitran/`

### 3.2 — Generating a Single Absorbance Spectrum
- Walkthrough of the simulation function:
  - Setting parameters: T (K), P (atm), ξ (mole fraction), L (cm), wavenumber range, step size
  - Using cached HITRAN line data for CH₄ (molecule 6, isotope 1)
  - Computing Voigt absorption coefficients
  - Converting to absorbance: α = coef × L × ξ
- Plotting ν vs. absorbance in Jupyter
- Saving to CSV with the naming convention: `TTT-P_P-LLLL-XXXXX.csv`

### 3.3 — Exploring Parameter Sensitivity
- Interactive notebook: vary T, P, and ξ individually and observe changes in line shape and peak height
- Pressure broadening effects (Voigt profile widening)
- Temperature effects on line strength via partition function changes
- Mole fraction scaling (linear in α)

**Deliverable:** Notebook that generates and visualizes absorbance spectra for arbitrary (T, P, ξ) conditions.

---

## Module 4: Simulating the Detector Voltage Trace

### 4.1 — Modeling the Laser Drive Signal
- Sinusoidal drive voltage: V(t) = A·cos(2πft) + offset, with V ∈ [0, 2] V
- Why a sinusoid instead of a triangle wave (smoother wavenumber tuning, baseline fitting is the CNN's job)
- Scan rate (f = 10 Hz), sample rate (250 kHz), samples per scan

### 4.2 — Mapping Voltage to Wavenumber Over Time
- The key assumption: dν/dt ∝ |dV/dt|
- Building the nonlinear ν(t) mapping using cumulative |sin| weighting
- The scan path: ν_max → ν_min → ν_max (one full period)
- Visualizing ν vs. t and comparing to Figure 2 from the documentation

### 4.3 — Computing Transmitted Intensity
- Nearest-neighbor lookup of absorbance at each ν(t) from the HITRAN-generated spectrum
- Applying Beer's Law: Iₜ(t) = I₀(t) · exp(−α(t))
- Plotting all four traces: I₀ vs t, ν vs t, α vs t, Iₜ vs t
- The "absorbance vs. wavelength (scan path)" view — understanding the doubled trace

### 4.4 — Vectorizing the Simulation for Speed
- Replacing the Python for-loop with `np.searchsorted` or vectorized `argmin`
- Timing comparison: loop vs. vectorized on a single scan
- This optimization is critical for generating 1M scans

**Deliverable:** Notebook producing a complete simulated detector trace from any (T, P, ξ) input.

---

## Module 5: Generating a One-Million-Scan Training Dataset

### 5.1 — Designing the Parameter Space
- Mole fraction range: 1–50,000 ppm (log-uniform or linear-uniform sampling)
- Temperature range: 253–323 K (−20°C to 50°C)
- Pressure range: 0.8–1.1 atm
- Random sampling strategy for (T, P, ξ) per scan
- Discussion: why 1M scans (vs. the original 50K), and the role of dataset size in CNN generalization

### 5.2 — Parallelizing with Multiprocessing
- Why this is CPU-bound (HITRAN Voigt calculations are NumPy, not GPU)
- Using Python `multiprocessing.Pool` with batch chunking
- Writing results incrementally to HDF5 (`h5py`) to manage memory
- HDF5 dataset structure: `nu`, `a` (or `I_t`), `T`, `P`, `xi`

### 5.3 — Running the Generation on the AMD Droplet
- Selecting an appropriate instance (CPU-heavy for generation, GPU for training)
- Monitoring progress with `tqdm`
- Spot-checking: randomly sample and plot a few scans to verify correctness
- Final dataset size and storage considerations

### 5.4 — Dataset Normalization Strategy
- Absorbance: global min-max normalization across all scans
- T and P: min-max to [0, 1]
- Mole fraction target: raw ppm (regression target)
- Why these choices matter for convergence

**Deliverable:** A `dataset_1M.h5` file containing 1M simulated scans ready for training.

---

## Module 6: Building the 1D-CNN in PyTorch

### 6.1 — Porting from TensorFlow/Keras to PyTorch
- Side-by-side comparison with the original Keras model (`train_cnn.py`)
- Defining the model class in PyTorch: `nn.Module` with `Conv1d`, `Dropout`, `Linear`
- Key difference: PyTorch expects `(batch, channels, length)` vs. Keras `(batch, length, channels)`
- Input shape: `(batch, 3, num_points)` — three channels for absorbance, T-feature, P-feature

### 6.2 — Model Architecture Walkthrough
- Conv1D(3→32, kernel=5) → ReLU → Conv1D(32→64, kernel=5) → ReLU → Dropout(0.2) → Flatten → Linear(128) → ReLU → Linear(1)
- Parameter count and receptive field discussion
- Why 1D convolutions are well-suited to spectral data (local feature extraction, translation invariance along the spectral axis)

### 6.3 — DataLoader Setup
- Custom `Dataset` class reading from HDF5
- Lazy loading vs. memory-mapped access for large datasets
- `DataLoader` with `num_workers`, `pin_memory`, and `shuffle`
- Train / validation / test split (70/15/15 or 80/10/10)

### 6.4 — Verifying GPU Utilization
- `.to(device)` for model and tensors
- Monitoring with `rocm-smi` during a training step
- Common pitfalls: data on CPU while model on GPU, small batch sizes underutilizing the GPU

**Deliverable:** A PyTorch model class and DataLoader, verified to run a single forward pass on the AMD GPU.

---

## Module 7: Training the Model

### 7.1 — Training Loop
- Loss function: MSE (regression)
- Optimizer: Adam, lr = 1e-3
- Writing a clean training loop with epoch/batch iteration
- Logging with `tqdm` progress bars in Jupyter

### 7.2 — Monitoring & Visualization
- Plotting train loss and validation loss per epoch (live-updating in Jupyter)
- Watching for overfitting: when val loss diverges from train loss
- Tracking MAE in ppm as a human-interpretable metric

### 7.3 — Hyperparameter Exploration
- Learning rate schedules: step decay, cosine annealing
- Batch size effects (64, 128, 256, 512) and GPU memory trade-offs
- Adding more Conv layers or increasing width
- Regularization: dropout rate, weight decay

### 7.4 — Training at Scale: Practical Considerations
- Estimated training time for 1M samples on an MI210/MI300X
- Checkpointing: saving model state every N epochs with `torch.save`
- Resuming from checkpoints
- Mixed precision training with `torch.amp` for faster throughput on AMD GPUs

**Deliverable:** A trained model checkpoint (`best_model.pt`) with plotted training curves showing convergence.

---

## Module 8: Inference — Predicting Mole Fraction from Unknown Scans

### 8.1 — Loading the Trained Model
- `model.load_state_dict(torch.load(...))` and setting `model.eval()`
- Moving to GPU for fast batch inference or CPU for single-scan inference

### 8.2 — Running Inference on a Single Scan
- Generating a new scan at known (T, P, ξ) that was NOT in the training set
- Preprocessing: same normalization pipeline used during training
- Forward pass → predicted mole fraction in ppm
- Comparing predicted vs. actual

### 8.3 — Evaluating Model Performance
- Running inference on the held-out test set
- Scatter plot: predicted vs. actual mole fraction
- Error distribution: histogram of (predicted − actual) in ppm and as % error
- Performance across the concentration range: does the model struggle at very low or very high ppm?
- MAE and RMSE summary statistics

### 8.4 — Stress-Testing the Model
- Extrapolation: what happens with ξ outside the training range?
- Noise injection: adding Gaussian noise to the detector trace to simulate real sensor noise
- Sensitivity to T and P errors: how wrong can the provided T, P be before predictions degrade?

**Deliverable:** Notebook demonstrating inference on unknown scans with error analysis.

---

## Module 9: Packaging & Next Steps

### 9.1 — Exporting the Model
- Saving with `torch.save` (state dict) and full model serialization
- Exporting to ONNX for deployment flexibility
- Brief mention of TorchScript for production inference

### 9.2 — Toward a Real Sensor
- What changes when you have real detector data (noise, baseline drift, etalon fringes)
- Transfer learning: fine-tuning the simulated model on small amounts of real data
- The role of the three CH₄ transitions in dynamic range (peak absorbances of ~19.7, ~5.5, ~2.1)

### 9.3 — Course Recap
- The full pipeline: HITRAN → absorbance → detector trace → dataset → 1D-CNN → inference
- Key takeaways: physics-informed data generation, PyTorch on ROCm, practical training at scale
- Suggested exercises: try a different molecule, vary path length, experiment with architectures (ResNet-1D, LSTM, Transformer)

---

## Appendices

### A — ROCm Troubleshooting
- Common issues: driver version mismatches, `ROCM_PATH` not set, PyTorch not detecting GPU
- Useful commands: `rocm-smi`, `rocminfo`, `hipconfig`

### B — HDF5 Dataset Schema
| Key   | Shape              | Description                       |
|-------|--------------------|-----------------------------------|
| `nu`  | (num_points,)      | Wavenumber grid (cm⁻¹)           |
| `a`   | (N, num_points)    | Absorbance spectra                |
| `T`   | (N,)               | Temperature (K)                   |
| `P`   | (N,)               | Pressure (atm)                    |
| `xi`  | (N,)               | Mole fraction (dimensionless)     |

### C — File Naming Convention
`TTT-P_P-LLLL-XXXXX.csv` — Temperature (K, 3 digits), Pressure (atm, decimal replaced with underscore), Path length (cm, 4 digits zero-padded), Mole fraction (ppm, 5 digits zero-padded).

### D — Session Setup Checklist
Every time you start a new droplet (fresh or from snapshot):
1. ☐ Access Jupyter via browser (http://[IP-ADDRESS])
2. ☐ Open `Module_01_Session_Setup.ipynb`
3. ☐ Run all cells (takes ~3-4 minutes)
4. ☐ Verify all checks pass (green ✅)
5. ☐ Ready to work on other modules!

# Group-Equivariant CNNs vs. standard CNNs

A project implementing an $SE(2)$ roto-translationally equivariant
convolutional neural network (**GE-CNN**) from steerable kernels, and comparing
it against a standard, structure-matched CNN on rotated Fashion-MNIST.

The GE-CNN builds rotational symmetry directly into the architecture, so that
the final classification is **invariant** to the rotation of the input while the
intermediate feature maps are **equivariant**. The project investigates three
questions:

1. **How well can roto-translational invariance be obtained using steerable basis functions?**
2. **How does network size affect a standard CNN vs. a GE-CNN, and how do they compare?**
3. **How does the training-set size affect a standard CNN vs. a GE-CNN, and how do they compare?**

## Repository structure

```txt
├── models/
│   ├── Model.py            # CNN and GE_CNN model definitions
│   ├── NN_layers.py        # Steerable layers: LiftingLayer, ConvLayer,
│   │                       #   ProjectionLayer, NormNonlinearity, etc.
│   └── model_weights/      # Saved weights used by the equivariance tests
├── src/
│   ├── data.py             # RotatedMNIST dataset + preprocessing (download, rotate, pad)
│   ├── train.py            # Generic train_loop / evaluate helpers
│   ├── test_parameters_datasize_v3.py   # Channel x datasize grid experiment (RQ2, RQ3)
│   ├── aggregate_param_datasize_v3.py   # Collect grid metrics into one array
│   ├── plot_param_datasize_2.py         # Plots for the grid experiment
│   ├── train_unrot_test_rot.py          # Train on unrotated, test on rotated (RQ1, OOD)
│   ├── train_untrot_test_unrot_and_test_rot.py  # Evaluate saved models on 3 test sets
│   ├── plot_train_unrot_test_rot_ci.py  # CI version of the OOD learning curve
│   ├── quantified_equivariance_test.py          # Quantify invariance/equivariance (unrotated training)
│   ├── quantified_equivariance_test_rotated.py  #   same, for models trained on rotated data
│   ├── seed_power_worst_average.py      # Power analysis: number of seeds needed
│   └── *.sh                # HPC (LSF) submit/run scripts
├── notebooks/              # Exploratory notebooks and figure generation
├── data/
│   ├── raw/                # Raw Fashion-MNIST (downloaded)
│   └── processed/          # Preprocessed .pt tensors (generated, git-ignored)
├── reports/                # Experiment outputs (metrics, plots)
├── results/                # Aggregated result tensors (.pt)
├── pyproject.toml          # Dependencies and project metadata
└── uv.lock                 # Locked dependency versions
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management and
requires Python 3.11–3.12.

```bash
# install all dependencies into a local virtual environment from uv.lock
uv sync
```

All commands below assume they are run from the repository root, either inside
the uv environment (`uv run python ...`) or an activated `.venv`.

## 1. Prepare the data

The dataset is Fashion-MNIST, downloaded via `kagglehub`. Running `data.py`
downloads the raw data, creates rotated and zero-padded ($28\times28 \to
40\times40$) versions, and saves everything as `.pt` tensors in
`data/processed/`. It also shows a sample of rotated images.

```bash
python src/data.py
```

This only needs to be run once. The processed tensors are git-ignored.

## 2. Run the experiments

### Equivariance and invariance (RQ1)

Quantifies how invariant the class scores and how equivariant the intermediate
feature maps are across 16 rotations, split into exact $90^\circ$ rotations
(no interpolation) and interpolated angles. The scripts load trained weights
from `models/model_weights/`.

```bash
python src/quantified_equivariance_test.py          # models trained on unrotated data
python src/quantified_equivariance_test_rotated.py  # models trained on rotated data
```

### Train unrotated, test rotated (out-of-distribution, RQ1)

Trains both models on unrotated data and evaluates on unrotated, exactly
$90^\circ$-rotated, and randomly rotated test sets.

```bash
python src/train_unrot_test_rot.py                  # produces reports/unrot_rot/
python src/train_untrot_test_unrot_and_test_rot.py  # evaluate saved models on 3 test sets
python src/plot_train_unrot_test_rot_ci.py          # learning curve with 95% CI
```

### Parameter and data-size grid (RQ2, RQ3)

A grid over channel width $\{2,4,8,16,32\}$ and training fraction
$\{0.5, 1, 3, 5, 10\}\%$, both models, 15 seeds, with early stopping. One job
trains a single grid cell, selected via the `TRAIN_FRACTION` and `CHANNELS`
environment variables, so the grid can be fanned out across HPC jobs.

```bash
# run a single cell locally (e.g. 10% of the data, 8 channels)
TRAIN_FRACTION=0.1 CHANNELS=8 python src/test_parameters_datasize_v3.py

# after all cells are done: aggregate and plot
python src/aggregate_param_datasize_v3.py   # -> reports/param_datasize_2/grid_results.pt
python src/plot_param_datasize_2.py         # -> reports/param_datasize_2/plots/
```

Per-cell results are written to
`reports/param_datasize_2/<MODEL>/datasize_<f>/channels_<c>/metrics.json`.

On the HPC cluster the grid is submitted with the LSF scripts
(`src/submit_param_datasize_v3.sh`, `src/run_param_datasize_v3.sh`).

### Power analysis (number of seeds)

Estimates the number of seeds needed to detect a 2-percentage-point difference
between the two models, given the variance of the paired per-seed differences.

```bash
python src/seed_power_worst_average.py
```

## Model overview

- **`models/Model.py`** — `CNN` (standard baseline) and `GE_CNN`. The baseline
  mirrors the GE-CNN's channel progression, depth, pooling and nonlinearity so
  the two are comparable in structure.
- **`models/NN_layers.py`** — the equivariant building blocks:
  - `fourier_basis` — circular-harmonic angular basis $e^{il\alpha}$ sampled on the kernel grid.
  - `LiftingLayer` — first layer, lifts the real image into the steerable basis (MLP radial profile).
  - `ConvLayer` — group convolution with steerable kernels (Gaussian-ring radial profile).
  - `NormNonlinearity` — norm-gated activation that preserves equivariance.
  - `ProjectionLayer` — reduces each feature map to rotation-invariant statistics (mean, max, std) for the final classification.
  - `ComplexAdaptiveAvgPool2d` — average pooling for complex-valued feature maps.

## Results

All experiment outputs (metrics and plots) are stored under `reports/`, and
aggregated result tensors under `results/`.

## Authors

Group 2:

| Name | Student no. | Email |
| --- | --- | --- |
| Uffe Grøn | s245109 | s245109@dtu.dk |
| Katinka Grønnegaard | s235058 | s235058@dtu.dk |
| Phi Vo | s245290 | s245290@dtu.dk |
| Lucas Burmester | s244322 | s244322@dtu.dk |

## License

See [LICENSE](LICENSE).

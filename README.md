# Image Representation Analysis

This project compares several image representations on CIFAR-10 using a simple 1-nearest-neighbor evaluation pipeline. It includes classic hand-crafted features, deep features from VGG-11, dataset utilities, feature caching, and a notebook that walks through the full analysis.

## What the Project Does

- Downloads and loads CIFAR-10 in `(N, C, H, W)` format
- Extracts and caches multiple feature representations
- Evaluates each representation with a 1-NN classifier
- Visualizes sample images and nearest-neighbor retrieval results
- Supports both pretrained and randomly initialized CNN feature extractors

## Representations Included

- `raw_pixel`: flattened image vectors
- `hog`: Histogram of Oriented Gradients descriptors
- `pretrained_cnn`:
  - `last_conv`
  - `last_fc`
- `random_cnn`:
  - `last_conv`
  - `last_fc`

## Repository Layout

```text
image-representation-analysis/
├── README.md
├── pyproject.toml
├── uv.lock
├── datasets/
├── features/
├── models/
│   └── vgg11_bn.pt
├── notebooks/
│   └── feature_analysis.ipynb
├── reports/
└── src/
    ├── __init__.py
    ├── dataset.py
    ├── extract_feature.py
    ├── models.py
    ├── path.py
    ├── vgg_network.py
    └── visualization.py
```

## Requirements

- Python `>=3.14`
- `torch`
- `torchvision`
- `numpy`
- `scikit-learn`
- `scikit-image`
- `matplotlib`
- `tqdm`
- `ipykernel`
- `huggingface-hub`
- `requests`
- `torchinfo`

Dependencies are defined in [pyproject.toml](pyproject.toml).

## Installation

Using `uv`:

```bash
uv sync
```

Using `pip`:

```bash
pip install -e .
```

## Running the Analysis

The main workflow lives in [notebooks/feature_analysis.ipynb](notebooks/feature_analysis.ipynb).

Start Jupyter:

```bash
uv run jupyter lab
```

Then open `notebooks/feature_analysis.ipynb`.

## Programmatic Usage

Load the dataset:

```python
from src.dataset import download_cifar10_dataset, load_dataset_splits

download_cifar10_dataset()
x_train, y_train, x_test, y_test = load_dataset_splits()
```

Extract or load cached features:

```python
from src.extract_feature import compute_or_load_features

raw_train, raw_test = compute_or_load_features(x_train, x_test, "raw_pixel")
hog_train, hog_test = compute_or_load_features(x_train, x_test, "hog")

pretrained_conv_train, pretrained_conv_test = compute_or_load_features(
    x_train, x_test, "pretrained_cnn", layer="last_conv"
)
```

Run 1-nearest-neighbor evaluation:

```python
from src.models import run_nearest_neighbor

classifier = run_nearest_neighbor(raw_train, y_train, raw_test, y_test)
```

Visualize examples and nearest neighbors:

```python
from src.visualization import visualize_cifar_data, visualize_nearest_neighbors

visualize_cifar_data(x_train.transpose(0, 2, 3, 1), y_train)
visualize_nearest_neighbors(
    x_test, y_test, x_train, y_train, classifier, raw_test, feature_name="raw_pixel"
)
```

## Data and Model Artifacts

- CIFAR-10 is downloaded into `datasets/cifar-10-batches-py/`
- Cached feature arrays are stored in `features/*.npz`
- A local CIFAR-10 VGG checkpoint is expected at `models/vgg11_bn.pt` when using functions from [src/vgg_network.py](src/vgg_network.py)

Feature caching is handled by `compute_or_load_features(...)`. If a matching `.npz` file already exists, the project loads it instead of recomputing features.

## Module Overview

[src/dataset.py](src/dataset.py)

- Downloads CIFAR-10
- Loads train/test splits
- Applies CIFAR-10 normalization statistics

[src/extract_feature.py](src/extract_feature.py)

- Computes raw pixel features
- Computes HOG features
- Extracts CNN features from VGG-11
- Saves and reloads cached feature files

[src/models.py](src/models.py)

- Trains and evaluates a 1-nearest-neighbor classifier

[src/visualization.py](src/visualization.py)

- Displays CIFAR-10 samples
- Shows correct and incorrect nearest-neighbor retrieval examples

[src/vgg_network.py](src/vgg_network.py)

- Defines a CIFAR-10-specific VGG model
- Loads a local pretrained checkpoint
- Evaluates the checkpoint on the test set

## Notes

- `src/extract_feature.py` uses `torchvision.models.vgg11_bn` for pretrained and random feature extraction.
- `src/vgg_network.py` defines a separate CIFAR-10 VGG implementation that expects a local checkpoint file.
- CNN feature extraction automatically uses CUDA when available and falls back to CPU otherwise.

## Output

During a typical run, you should expect:

- printed dataset shapes
- printed feature matrix shapes
- cached `.npz` feature files under `features/`
- 1-NN accuracy values on the CIFAR-10 test split
- matplotlib figures for samples and nearest-neighbor matches

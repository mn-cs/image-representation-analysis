# CIFAR-10 Feature Representation Analysis

A comprehensive study comparing different image feature representations for visual recognition tasks. This project implements and evaluates multiple feature extraction methods including raw pixels, traditional computer vision features (HOG), and deep learning representations (CNN features) on the CIFAR-10 dataset.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/micben-cs/image-representation.git
cd image-representation

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .

# Run the main analysis
python main.py

# Or explore interactively
jupyter lab feature_analysis.ipynb
```

## Features & Capabilities

- **Multiple Feature Extraction Methods**: Raw pixels, HOG descriptors, and CNN features
- **Pretrained Model Integration**: VGG-11 with batch normalization for feature extraction
- **Comprehensive Evaluation**: k-NN classification across different representation spaces
- **Visualization Tools**: Dataset exploration and feature space analysis
- **Modular Architecture**: Easy to extend with additional feature extractors

### Supported Representations

1. **Raw Pixel Features** - Direct pixel intensity vectors
2. **HOG Features** - Histogram of Oriented Gradients for edge/texture capture
3. **CNN Features** - Deep representations from VGG-11 at multiple levels
   - Intermediate layer activations
   - Final layer features
   - Comparison with randomly initialized networks

## Project Structure

```
cifar-representations/
├── feature_analysis.ipynb   # Interactive analysis and experiments
├── main.py                  # Main execution script
├── dataset.py               # CIFAR-10 data loading and preprocessing
├── extract_feature.py       # Feature extraction implementations
├── vgg_network.py           # VGG-11 neural network architecture
├── visual_nn_results.py     # Results visualization utilities
├── path.py                  # Path configuration management
├── pyproject.toml           # Project dependencies and metadata
├── uv.lock                  # Dependency lock file
├── models/                  # Pretrained model weights
│   └── vgg11_bn.pt          # VGG-11 with batch normalization
└── datasets/                # CIFAR-10 dataset (auto-downloaded)
    └── cifar-10-batches-py/ # Raw CIFAR-10 data files
```

````

## Requirements

- **Python**: 3.12+
- **Core Dependencies**:
  - PyTorch 2.8+ (CPU sufficient, GPU optional for faster processing)
  - scikit-learn 1.7+ for machine learning utilities
  - scikit-image for HOG feature extraction
  - NumPy, Matplotlib for numerical computing and visualization
  - Jupyter Lab/Notebook for interactive analysis

All dependencies are managed via `pyproject.toml` with locked versions in `uv.lock` for reproducible builds.

## Installation

### Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup the project
git clone https://github.com/micben-cs/image-representation.git
cd image-representation
uv sync
```

### Using pip

```bash
git clone https://github.com/micben-cs/image-representation.git
cd image-representation
pip install -e .
```

## Usage

### Quick Start

1. **Download CIFAR-10 dataset:**

   ```python
   from dataset import download_cifar10_dataset
   download_cifar10_dataset()
   ```

2. **Load the dataset:**

   ```python
   from dataset import load_dataset_splits
   x_train, y_train, x_test, y_test = load_dataset_splits()
   ```

3. **Extract features:**
   ```python
   from extract_feature import compute_raw_pixel_features, compute_hog_features
   # Raw pixel features
   raw_train, raw_test = compute_raw_pixel_features(x_train, x_test)
   # HOG features
   hog_train, hog_test = compute_hog_features(x_train, x_test)
   ```

### Running the Analysis

**Command Line Interface:**
```bash
python main.py
```

**Interactive Analysis:**
```bash
jupyter lab feature_analysis.ipynb
```

The analysis pipeline includes:
- Automatic CIFAR-10 dataset downloading and preprocessing
- Feature extraction across multiple representation methods
- k-NN classification evaluation with performance metrics
- Visualization of results and feature spaces

## Key Functions

### Dataset Operations (`dataset.py`)

- `download_cifar10_dataset()` - Downloads CIFAR-10 dataset
- `load_dataset_splits()` - Loads train/test splits
- `visualize_cifar_data()` - Displays sample images from each class

### Feature Extraction (`extract_feature.py`)

- `compute_raw_pixel_features()` - Flattens images to pixel vectors
- `compute_hog_features()` - Extracts HOG descriptors
- `compute_cnn_features()` - Extracts CNN features from VGG-11

### VGG Network (`vgg_network.py`)

- `VGG` class - VGG-11 architecture for CIFAR-10
- `vgg11_bn()` - Creates VGG-11 with batch normalization
- `load_pretrained_vgg()` - Loads pretrained weights

## Performance Considerations

- **Hardware Requirements**: Runs efficiently on CPU-only systems; GPU acceleration optional
- **Memory Usage**: CNN feature extraction may require 4-8GB RAM for full dataset processing
- **Processing Time**: Full feature extraction typically completes within 10-30 minutes on modern hardware

## Methodology

This project implements a systematic comparison of image representation methods:

1. **Data Pipeline**: Automated CIFAR-10 download, preprocessing, and normalization
2. **Feature Extraction**: Multiple representation methods with consistent preprocessing
3. **Classification**: k-nearest neighbor evaluation across feature spaces
4. **Evaluation**: Comprehensive performance metrics and statistical analysis
5. **Visualization**: Feature space analysis and results interpretation

## Results

The project demonstrates how different image representations affect classification performance:

- Raw pixels provide baseline performance
- HOG features capture edge and texture information
- Pretrained CNN features leverage learned representations
- Random CNN features show the importance of training

## Contributing

We welcome contributions! Please feel free to:

- Submit bug reports and feature requests via GitHub Issues
- Propose new feature extraction methods
- Improve documentation and examples
- Add support for additional datasets

## Citation

If you use this code in your research, please cite:

```bibtex
@software{image_representation_analysis,
  title={Image Representation Analysis},
  author={micben-cs},
  year={2025},
  url={https://github.com/micben-cs/image-representation}
}
```

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [VGG Architecture](https://arxiv.org/abs/1409.1556)
- [HOG Features](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients)
- [PyTorch CIFAR-10 VGG Implementation](https://github.com/huyvnphan/PyTorch_CIFAR10)

## License

This project is for educational purposes. Dataset and model weights follow their respective licenses.
````

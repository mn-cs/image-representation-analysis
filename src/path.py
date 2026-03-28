from pathlib import Path

# Get the directory containing this file (__file__)
src_path = Path(__file__).resolve().parent

# Go up one level to get the project root
root_path = src_path.parent

# Use the forward slash '/' operator to cleanly join paths
dataset_path = root_path / "datasets"
cifar10_path = dataset_path / "cifar-10-batches-py"
feature_path = root_path / "features"
model_path = root_path / "models"
figure_path = root_path / "figures"

visualize_nn_results_path = src_path / "visual_nn_results.py"
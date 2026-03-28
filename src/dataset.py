import pickle
import os
import numpy as np
import urllib.request
import tarfile
from src.path import dataset_path, cifar10_path

import warnings
warnings.filterwarnings('ignore')

CIFAR_WIDTH = 32
CIFAR_HEIGHT = 32
CIFAR_CHANNEL = 3


def download_cifar10_dataset():
    """Download CIFAR-10 if not already present."""
    if cifar10_path.exists():
        print("✓ CIFAR-10 dataset already exists in ./../datasets/cifar-10-batches-py Skipping download.")
        return

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    dataset_path.mkdir(parents=True, exist_ok=True)

    print("Downloading CIFAR-10 dataset...")
    fileobj = urllib.request.urlopen(url)

    with tarfile.open(fileobj=fileobj, mode="r|gz") as tar:
        tar.extractall(path=dataset_path)

    print('✓ CIFAR-10 dataset downloaded successfully to ./../datasets/cifar-10-batches-py')


def load_dataset_splits():
    """Load CIFAR-10 and return train/test splits."""
    dataset = load_cifar10_dataset()
    print("======> CIFAR-10 dataset loaded")

    print("Training set data shape:", dataset['x_train'].shape)
    print("Training set label shape:", dataset['y_train'].shape)
    print("Test set data shape:", dataset['x_test'].shape)
    print("Test set label shape:", dataset['y_test'].shape)

    x_train = dataset['x_train']
    y_train = dataset['y_train']
    x_test = dataset['x_test']
    y_test = dataset['y_test']

    return x_train, y_train, x_test, y_test


def load_one_cifar_batch(file_name: str):
    """Load a single batch of CIFAR-10 data."""
    with open(file_name, 'rb') as f:
        batch_data = pickle.load(
            f, encoding='bytes'
        )
        batch_data[b"data"] = batch_data[b"data"]
    
        return batch_data[b"data"], batch_data[b"labels"]


def load_cifar10_dataset(dataset_path: str = cifar10_path, subset_train: int = 50000, subset_test: int = 10000):
    """Load CIFAR-10 dataset and return training and test splits."""
    x_train = []
    y_train = []
    for i in range(1, 6):
        x_batch, y_batch = load_one_cifar_batch(
            os.path.join(dataset_path, "data_batch_{}".format(i))
        )
        x_train.append(x_batch)
        y_train.append(y_batch)

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_test, y_test = load_one_cifar_batch( os.path.join(dataset_path, "test_batch"))
    y_test = np.array(y_test)

    dataset = {
        "x_train": x_train[:subset_train],
        "y_train": y_train[:subset_train],
        "x_test": x_test[:subset_test],
        "y_test": y_test[:subset_test]
    }

    dataset["x_train"] = dataset["x_train"].reshape(
        (-1, CIFAR_CHANNEL, CIFAR_WIDTH, CIFAR_HEIGHT)
    )
    dataset["x_test"] = dataset["x_test"].reshape(
        (-1, CIFAR_CHANNEL, CIFAR_WIDTH, CIFAR_HEIGHT)
    )

    return dataset


def get_cifar10_mu_std_img():
    """Get mean and standard deviation images for CIFAR-10 normalization."""
    mu = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])

    mu_img = np.zeros((CIFAR_CHANNEL, CIFAR_WIDTH, CIFAR_HEIGHT), dtype=np.float32)
    std_img = np.zeros((CIFAR_CHANNEL, CIFAR_WIDTH, CIFAR_HEIGHT), dtype=np.float32)
    
    for i in range(mu.shape[0]):
        mu_img[i, ...] = mu[i]
        std_img[i, ...] = std[i]
    
    return mu_img, std_img
    


def normalize(X, mu = None, std = None):
    """Normalize CIFAR-10 images using mean and standard deviation."""
    X /= 255.0
    
    if std is None:
        std = np.std(X, axis =0)
    if mu is None:
        mu = np.mean(X, axis =0)
    
    return (X - mu) / std
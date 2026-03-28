import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from skimage.feature import hog
from torchvision.models import VGG11_BN_Weights, vgg11_bn
from tqdm import tqdm

from src.dataset import get_cifar10_mu_std_img, normalize
from src.path import feature_path


def save_features(train_features, test_features, sp_feature_path):
    """Save train and test features to a compressed .npz file."""
    sp_feature_path = Path(sp_feature_path)
    sp_feature_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        sp_feature_path,
        train=train_features,
        test=test_features,
    )

    print(f"======> Saved train and test features to ./../features/{sp_feature_path.name}")


def load_features(sp_feature_path):
    """Load train and test features from a compressed .npz file."""
    sp_feature_path = Path(sp_feature_path)
    features = np.load(sp_feature_path, allow_pickle=False)
    
    print(f"======> Loaded train and test features from ./../features/{sp_feature_path.name}")

    return features["train"], features["test"]


def compute_raw_pixel_features(x_train, x_test):
    """Compute raw pixel features by flattening the images."""
    raw_pixel_train_features = x_train.reshape(x_train.shape[0], -1)
    raw_pixel_test_features = x_test.reshape(x_test.shape[0], -1)

    print("======> Done with computation of raw pixel features")
    return raw_pixel_train_features, raw_pixel_test_features


def compute_hog_features(x_train, x_test):
    """Compute Histogram of Oriented Gradients (HoG) features."""
    num_train_samples = x_train.shape[0]
    num_test_samples = x_test.shape[0]

    hog_train_features = []
    for i in tqdm(range(num_train_samples)):
        fd = hog(
            x_train[i],
            orientations=8,
            pixels_per_cell=(4, 4),
            cells_per_block=(1, 1),
            feature_vector=True,
            channel_axis=0,
        )
        hog_train_features.append(fd)

    hog_test_features = []
    for i in tqdm(range(num_test_samples)):
        fd = hog(
            x_test[i],
            orientations=8,
            pixels_per_cell=(4, 4),
            cells_per_block=(1, 1),
            feature_vector=True,
            channel_axis=0,
        )
        hog_test_features.append(fd)

    hog_train_features = np.asarray(hog_train_features)
    hog_test_features = np.asarray(hog_test_features)

    print("======> Done with computation of HoG features")
    return hog_train_features, hog_test_features


def compute_pretrained_cnn_features(x_train, x_test, mu_img, std_img, layer):
    """Compute features using a pretrained CNN."""
    deep_model = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
    deep_model.eval()

    return compute_cnn_features(
        deep_model=deep_model,
        x_train=x_train,
        x_test=x_test,
        mu_img=mu_img,
        std_img=std_img,
        layer=layer,
    )


def compute_random_cnn_features(x_train, x_test, mu_img, std_img, layer):
    """Compute features using a randomly initialized CNN."""
    deep_model = vgg11_bn(weights=None)
    deep_model.eval()

    return compute_cnn_features(
        deep_model=deep_model,
        x_train=x_train,
        x_test=x_test,
        mu_img=mu_img,
        std_img=std_img,
        layer=layer,
    )


def _extract_features_in_batches(deep_model, x_data, layer, batch_size, device):
    """Extract features from a deep model in batches."""
    num_samples = x_data.shape[0]
    feature_batches = []

    feature_extractor = nn.Sequential(*list(deep_model.classifier.children())[:-1]).to(device)
    feature_extractor.eval()

    with torch.no_grad():
        for start_idx in tqdm(range(0, num_samples, batch_size)):
            end_idx = min(start_idx + batch_size, num_samples)

            x_batch_np = x_data[start_idx:end_idx]
            x_batch = torch.from_numpy(x_batch_np).to(device=device, dtype=torch.float32)

            x_feat = deep_model.features(x_batch)
            x_feat = deep_model.avgpool(x_feat)
            x_feat = torch.flatten(x_feat, 1)

            if layer == "last_conv":
                cur_feature_batch = x_feat.cpu().numpy()
            elif layer == "last_fc":
                cur_feature_batch = feature_extractor(x_feat).cpu().numpy()
            else:
                raise ValueError(f"Unsupported layer: {layer}")

            feature_batches.append(cur_feature_batch)

    return np.concatenate(feature_batches, axis=0)


def compute_cnn_features(deep_model, x_train, x_test, mu_img, std_img, layer, batch_size=100):
    """Compute features using a CNN model."""
    if layer not in ["last_conv", "last_fc"]:
        raise ValueError("layer must be one of: ['last_conv', 'last_fc']")

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    x_train_ = normalize(np.copy(x_train).astype(np.float32), mu_img, std_img)
    x_test_ = normalize(np.copy(x_test).astype(np.float32), mu_img, std_img)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deep_model = deep_model.to(device)
    deep_model.eval()

    x_deep_features_train = _extract_features_in_batches(
        deep_model=deep_model,
        x_data=x_train_,
        layer=layer,
        batch_size=batch_size,
        device=device,
    )

    x_deep_features_test = _extract_features_in_batches(
        deep_model=deep_model,
        x_data=x_test_,
        layer=layer,
        batch_size=batch_size,
        device=device,
    )

    print("======> Done with computation of CNN features")
    return x_deep_features_train, x_deep_features_test


def compute_or_load_features(x_train, x_test, feature_type, layer=None):
    """Load features from disk or compute and save them."""
    valid_feature_types = ["raw_pixel", "hog", "pretrained_cnn", "random_cnn"]
    if feature_type not in valid_feature_types:
        raise ValueError(f"feature_type must be one of: {valid_feature_types}")

    if feature_type in ["raw_pixel", "hog"]:
        if layer is not None:
            raise ValueError(
                "layer can only be set when feature_type is 'pretrained_cnn' or 'random_cnn'"
            )
    else:
        if layer not in ["last_conv", "last_fc"]:
            raise ValueError("layer must be one of: ['last_conv', 'last_fc']")

    if layer is None:
        sp_feature_path = os.path.join(feature_path, f"{feature_type}.npz")
    else:
        sp_feature_path = os.path.join(feature_path, f"{feature_type}_{layer}.npz")

    feature_file = Path(sp_feature_path)

    if feature_file.is_file():
        train_features, test_features = load_features(sp_feature_path)
    else:
        if feature_type == "raw_pixel":
            train_features, test_features = compute_raw_pixel_features(x_train, x_test)
        elif feature_type == "hog":
            train_features, test_features = compute_hog_features(x_train, x_test)
        elif feature_type == "pretrained_cnn":
            mu_img, std_img = get_cifar10_mu_std_img()
            train_features, test_features = compute_pretrained_cnn_features(
                x_train, x_test, mu_img, std_img, layer
            )
        elif feature_type == "random_cnn":
            mu_img, std_img = get_cifar10_mu_std_img()
            train_features, test_features = compute_random_cnn_features(
                x_train, x_test, mu_img, std_img, layer
            )
        else:
            raise NotImplementedError

        save_features(train_features, test_features, sp_feature_path)

    print("Training feature shape:", train_features.shape)
    print("Test feature shape:", test_features.shape)

    return train_features, test_features
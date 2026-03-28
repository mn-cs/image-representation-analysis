import numpy as np
import torch
import torch.nn as nn

from src.dataset import get_cifar10_mu_std_img, normalize
from src.path import model_path

__all__ = ["VGG", "vgg11_bn"]


class VGG(nn.Module):
    """VGG model adapted for CIFAR-10 classification. The architecture is based on the original VGG paper, but modified to work with 32x32 input images and 10 output classes."""
    def __init__(self, features, num_classes=10, init_weights=True):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def extract_features(self, x, layer):
        """Extract features from a specific layer of the VGG model."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if layer == "last_conv":
            return x
        if layer == "last_fc":
            return self.classifier[:-1](x)

        raise ValueError("layer must be one of: ['last_conv', 'last_fc']")

    def forward(self, x):
        """Forward pass of the VGG model."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Initialize the weights of the VGG model."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3

    for value in cfg:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, value, kernel_size=3, padding=1)
            if batch_norm:
                layers.extend([conv2d, nn.BatchNorm2d(value), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_channels = value

    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
}


def ensure_vgg11_bn_weights():
    weights_path = model_path / "vgg11_bn.pt"
    if weights_path.exists():
        return weights_path

    raise FileNotFoundError(
        f"Missing pretrained weights: {weights_path}\n"
        "Place the CIFAR-10 pretrained vgg11_bn.pt file in the models/ folder."
    )


def _vgg(arch, cfg, batch_norm, pretrained=False, device="cpu", **kwargs):
    """Create a VGG model with the specified configuration."""
    if pretrained:
        kwargs["init_weights"] = False

    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    if pretrained:
        weights_path = ensure_vgg11_bn_weights()
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

    model = model.to(device)
    return model


def vgg11_bn(pretrained=False, device="cpu", **kwargs):
    """VGG-11 model with batch normalization."""
    return _vgg("vgg11_bn", "A", True, pretrained=pretrained, device=device, **kwargs)


def test_pretrained_vgg(x_test, y_test, batch_size=100, device=None): 
    """Test the pretrained VGG model on the CIFAR-10 test set."""   
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    vgg_network = vgg11_bn(pretrained=True, device=device)
    vgg_network.eval()

    mu_img, std_img = get_cifar10_mu_std_img()
    x_test_normalized = normalize(np.copy(x_test).astype(np.float32), mu_img, std_img)

    num_test_samples = x_test_normalized.shape[0]
    correct = 0
    total = 0

    with torch.no_grad():
        for start_idx in range(0, num_test_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_test_samples)

            x_batch = torch.from_numpy(x_test_normalized[start_idx:end_idx]).to(
                device=device, dtype=torch.float32
            )
            y_batch = torch.from_numpy(y_test[start_idx:end_idx]).to(device=device)

            outputs = vgg_network(x_batch)
            predicted_labels = outputs.argmax(dim=1)

            total += y_batch.size(0)
            correct += (predicted_labels == y_batch).sum().item()

    return correct / total
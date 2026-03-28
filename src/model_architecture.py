"""CNN Model Architecture Module -- PyTorch nn.Module implementations."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CINIC_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        # 32x32 -> 16x16 -> 8x8 -> 4x4; 128 channels -> 128*4*4 = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def create_baseline_cnn(input_shape=(32, 32, 3), num_classes=10):
    return BaselineCNN(num_classes=num_classes)


class DeepCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        # 32->16->8->4->2; 256 channels -> 256*2*2 = 1024
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def create_deep_cnn(input_shape=(32, 32, 3), num_classes=10):
    return DeepCNN(num_classes=num_classes)


class EfficientCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def create_efficient_cnn(input_shape=(32, 32, 3), num_classes=10):
    return EfficientCNN(num_classes=num_classes)


class CNNWithRegularization(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(dropout_rate * 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def create_cnn_with_regularization(
    input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.3, weight_decay=1e-4
):
    # weight_decay is NOT stored on the model.
    # Pass it to the optimizer: torch.optim.Adam(model.parameters(), weight_decay=wd)
    return CNNWithRegularization(num_classes=num_classes, dropout_rate=dropout_rate)


class FewShotCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def create_few_shot_cnn(input_shape=(32, 32, 3), num_classes=10):
    return FewShotCNN(num_classes=num_classes)


# ── VGG-style baseline ────────────────────────────────────────────────────────

def _conv_bn_relu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGGBaseline(nn.Module):
    """VGG-style baseline: all 3×3 conv, BN+ReLU on every conv, 3 pool stages.

    Block 1 (3 convs): 3 → 32 → 32 → 32,  MaxPool → 16×16
    Block 2 (3 convs): 32 → 64 → 64 → 64, MaxPool → 8×8
    Block 3 (2 convs): 64 → 128 → 128,     MaxPool → 4×4
    Classifier: Flatten(2048) → Linear(512) → BN → ReLU → Dropout(0.5) → Linear(10)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            _conv_bn_relu(3, 32), _conv_bn_relu(32, 32), _conv_bn_relu(32, 32),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            # Block 2
            _conv_bn_relu(32, 64), _conv_bn_relu(64, 64), _conv_bn_relu(64, 64),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            # Block 3
            _conv_bn_relu(64, 128), _conv_bn_relu(128, 128),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        # 32 → 16 → 8 → 4;  128 × 4 × 4 = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def create_vgg_baseline(input_shape=(32, 32, 3), num_classes=10):
    return VGGBaseline(num_classes=num_classes)


# ── ResNet-style deep model ───────────────────────────────────────────────────

class BasicResidualBlock(nn.Module):
    """Two-layer residual block with optional projection shortcut.

    When stride > 1 or in_ch != out_ch a 1×1 conv projects the skip connection
    to matching dimensions (same approach as ResNet-34 option B).
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


def _make_stage(in_ch, out_ch, num_blocks, stride=1):
    blocks = [BasicResidualBlock(in_ch, out_ch, stride=stride)]
    for _ in range(1, num_blocks):
        blocks.append(BasicResidualBlock(out_ch, out_ch, stride=1))
    return nn.Sequential(*blocks)


class ResNetDeep(nn.Module):
    """CIFAR-style ResNet with 9 residual blocks across 3 stages.

    Stem : Conv(3→64, 3×3) + BN + ReLU  — no MaxPool (images are only 32×32)
    Stage 1: 3 × BasicResidualBlock(64→64,  stride=1) → 32×32
    Stage 2: 3 × BasicResidualBlock(64→128, stride=2) → 16×16
    Stage 3: 3 × BasicResidualBlock(128→256,stride=2) → 8×8
    Head : GlobalAvgPool → Flatten → Dropout(0.3) → Linear(256→10)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = _make_stage(64,  64,  num_blocks=3, stride=1)
        self.stage2 = _make_stage(64,  128, num_blocks=3, stride=2)
        self.stage3 = _make_stage(128, 256, num_blocks=3, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


def create_resnet_deep(input_shape=(32, 32, 3), num_classes=10):
    return ResNetDeep(num_classes=num_classes)


def get_model_summary(model, model_name="Model"):
    print(f"\n{model_name} Architecture Summary:")
    print("=" * 50)
    try:
        import torchinfo
        torchinfo.summary(model, input_size=(1, 3, 32, 32))
    except ImportError:
        print(model)

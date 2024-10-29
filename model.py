from torcheeg.models import *
import torch.nn as nn
import torch.nn.functional as F
from models.cbam import CBAMBlock
from utils import set_seed


# self-defined model
class CBAMFeatureMapping(nn.Module):
    class ConvBN(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding
            )
            self.bn = nn.BatchNorm2d(out_channels)

        def forward(self, x):
            return F.relu(self.bn(self.conv(x)))

    class ResBlock(nn.Module):
        def __init__(self, num_channels):
            super().__init__()
            self.convbn1 = CBAMFeatureMapping.ConvBN(
                num_channels, num_channels, 3, 1, 1
            )
            self.convbn2 = CBAMFeatureMapping.ConvBN(
                num_channels, num_channels, 3, 1, 1
            )
            self.convbn3 = CBAMFeatureMapping.ConvBN(
                num_channels, num_channels, 3, 1, 1
            )

        def forward(self, x):
            out = self.convbn1(x)
            out = self.convbn2(out)
            residual = self.convbn3(x)
            return out + residual

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.resblock = self.ResBlock(self.in_channels)
        self.cbam = CBAMBlock(self.in_channels)  # CBAM imported from cbam.py
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(
            in_features=self.in_channels * 4 * 4, out_features=self.num_classes
        )

    def forward(self, x):
        x = self.resblock(x)
        x = self.cbam(x)
        x = self.bn(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.fc(x)
        return x


NUM_CLASSES = 2
IN_CHANNELS = 4

def load_model(model_name, seed):
    # reproducibility
    set_seed(seed)
    
    # load model based on model_name
    if model_name == "fbccnn":
        return FBCCNN(
            num_classes=NUM_CLASSES, in_channels=IN_CHANNELS, grid_size=(9, 9)
        )
    elif model_name == "ccnn":
        return CCNN(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS, grid_size=(9, 9))
    elif model_name == "simplevit":
        return SimpleViT(
            chunk_size=IN_CHANNELS,
            grid_size=(9, 9),
            t_patch_size=1,
            num_classes=NUM_CLASSES,
        )
    elif model_name == "vit":
        return ViT(
            chunk_size=IN_CHANNELS,
            grid_size=(9, 9),
            t_patch_size=1,
            num_classes=NUM_CLASSES,
        )
    elif model_name == "cbamfm":
        return CBAMFeatureMapping(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    else:
        return None

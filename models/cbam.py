import torch
import torch.nn as nn

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pooled = self.avg_pool(x).view(x.size(0), -1)
        max_pooled = self.max_pool(x).view(x.size(0), -1)
        avg_out = self.fc(avg_pooled)
        max_out = self.fc(max_pooled)
        out = self.sigmoid(avg_out + max_out)
        return out.view(x.size(0), x.size(1), 1, 1)

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pooled = torch.mean(x, dim=1, keepdim=True)
        max_pooled, _ = torch.max(x, dim=1, keepdim=True)
        concatenated = torch.cat([avg_pooled, max_pooled], dim=1)
        out = self.conv1(concatenated)
        return self.sigmoid(out)

class CBAMBlock(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(num_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
from __future__ import annotations
import torch 
import torch.nn as nn

class AlexNet64Gray(nn.Module):
    """AlexNet-Style CNN adapted for 1x64x64 grayscale images.

    Args:
        nn (nn.Module): Basic inheritance from neural network

    Feature maps:
        Input :         1   x 64 x 64
        Conv1 :         64  x 64 x 64 (5x5, stride = 1, pad = 2) + ReLU
        Pool1 :         64  x 32 x 32 (2x2, stride = 2)          + ReLU
        Conv2 :         192 x 32 x 32 (3x3, stride = 1, pad = 1) + ReLU
        Pool2 :         192 x 16 x 16 (2x2, stride = 2)          + ReLU
        Conv3 :         384 x 16 x 16 (3x3, stride = 1, pad = 1) + ReLU
        Conv4 :         256 x 16 x 16 (3x3, stride = 1, pad = 1) + ReLU
        Conv5 :         256 x 16 x 16 (3x3, stride = 1, pad = 1) + ReLU
        Pool3 :         256 x 8  x 8  (2x2, stride = 2)          + ReLU
        Flatten : 16384
        FC6   :         1024
        FC7   :         1024
        FC8   :         num_classes (default = 10)
    """
    def __init__(
            self,
            num_classes: int = 10,
            in_channels: int = 1,
            conv1_out: int = 64,
            conv2_out: int = 192,
            conv3_out: int = 384,
            conv4_out: int = 256,
            conv5_out: int = 256,
            fc6_out: int = 1024,
            fc7_out: int = 1024,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, conv1_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(conv1_out, conv2_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(conv2_out, conv3_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(conv3_out, conv4_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(conv4_out, conv5_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self._flattened_size = conv5_out * 8 * 8

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flattened_size, fc6_out, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(fc6_out, fc7_out, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(fc7_out, num_classes, bias=True),
        )

        self.num_classes = num_classes

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
    
def number_of_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
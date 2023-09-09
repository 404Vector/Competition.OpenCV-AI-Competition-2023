import torch
from torch import nn
import timm


class ColorRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("resnet34", pretrained=True, num_classes=3)

    def forward(self, x):
        x = self.model(x)
        return x

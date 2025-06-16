import numpy as np
import torch

from torch import nn


class PeakSignalToNoiseRatio(nn.Module):
    def __init__(self, max_val=1.0):
        super(PeakSignalToNoiseRatio, self).__init__()
        self.mse = nn.MSELoss()

        self.max_val = max_val

    def forward(self, a, b):
        return 20 * np.log10(self.max_val) - 10 * torch.log10(self.mse(a, b))

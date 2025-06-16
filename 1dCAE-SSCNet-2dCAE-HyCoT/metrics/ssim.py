import pytorch_msssim

from torch import nn


class StructuralSimilarity(nn.Module):
    def __init__(self, data_range=1.0, channels=103):
        super(StructuralSimilarity, self).__init__()
        self.ssim = pytorch_msssim.SSIM(data_range=data_range, channel=channels)

    def forward(self, a, b):
        return self.ssim(a, b)

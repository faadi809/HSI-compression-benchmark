import pytorch_msssim

from torch import nn


class StructuralSimilarity(nn.Module):
    def __init__(self, data_range=1.0, channels=202):
        super(StructuralSimilarity, self).__init__()
        self.ssim = pytorch_msssim.MS_SSIM(data_range=data_range, channel=channels)

    def forward(self, a, b):
        return self.ssim(a, b)
    

if __name__ == "__main__":
    import torch

    img1 = torch.rand(1, 202, 128, 128)
    img2 = torch.rand(1, 202, 128, 128)

    msssim = StructuralSimilarity()

    print(msssim(img1, img2))


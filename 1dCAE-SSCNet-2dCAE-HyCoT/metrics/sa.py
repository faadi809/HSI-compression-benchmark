import torch

from torch import nn


class SpectralAngle(nn.Module):
    def __init__(self):
        super(SpectralAngle, self).__init__()

    @staticmethod
    def forward(a, b):
        numerator = torch.sum(a * b, dim=1)
        denominator = torch.sqrt(torch.sum(a ** 2, dim=1) * torch.sum(b ** 2, dim=1))
        fraction = numerator / denominator
        sa = torch.acos(fraction)
        sa_degrees = torch.rad2deg(sa)
        sa_degrees = torch.mean(sa_degrees)
        return sa_degrees

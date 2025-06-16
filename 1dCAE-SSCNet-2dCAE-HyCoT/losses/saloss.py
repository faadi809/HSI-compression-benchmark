from torch import nn

from metrics import sa


class SpectralAngleLoss(nn.Module):
    def __init__(self):
        super(SpectralAngleLoss, self).__init__()
        self.metric = sa.SpectralAngle()

    def forward(self, x, x_hat):
        return self.metric(x, x_hat)

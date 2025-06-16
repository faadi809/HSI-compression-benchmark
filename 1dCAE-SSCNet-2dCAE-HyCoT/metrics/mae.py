from torch import nn


class MeanAbsoluteError(nn.Module):
    def __init__(self):
        super(MeanAbsoluteError, self).__init__()
        self.mae = nn.L1Loss()

    def forward(self, a, b):
        return self.mae(a, b)

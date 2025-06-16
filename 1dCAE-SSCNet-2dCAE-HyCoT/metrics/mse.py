from torch import nn


class MeanSquaredError(nn.Module):
    def __init__(self):
        super(MeanSquaredError, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, a, b):
        return self.mse(a, b)

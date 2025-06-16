from torch import nn

from metrics import mse


class MeanSquaredErrorLoss(nn.Module):
    def __init__(self):
        super(MeanSquaredErrorLoss, self).__init__()
        self.metric = mse.MeanSquaredError()

    def forward(self, x, x_hat):
        return self.metric(x, x_hat)

from torch import nn

from metrics import mae


class MeanAbsoluteErrorLoss(nn.Module):
    def __init__(self):
        super(MeanAbsoluteErrorLoss, self).__init__()
        self.metric = mae.MeanAbsoluteError()

    def forward(self, x, x_hat):
        return self.metric(x, x_hat)

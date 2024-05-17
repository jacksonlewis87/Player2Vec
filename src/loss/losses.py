import torch

from enum import Enum


class Loss(Enum):
    L2 = "l2"
    BCE = "bce"


class L2Loss(torch.nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, output, labels):
        labels_t = torch.transpose(labels, 0, 1)
        output_t = torch.transpose(output, 0, 1)
        return torch.mean(torch.sum(torch.pow(labels_t - output_t, 2), dim=0))


class WeightedL2Loss(torch.nn.Module):
    def __init__(self):
        super(WeightedL2Loss, self).__init__()
        self.softmax = torch.nn.Softmax()

    def forward(self, output, labels, weights):
        labels_t = torch.transpose(labels, 0, 1)
        output_t = torch.transpose(output, 0, 1)
        return torch.sum(torch.mean(torch.pow(labels_t - output_t, 2), dim=1) * self.softmax(weights))


class BCELoss(torch.nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, output, labels):
        loss = labels * torch.log(output + 1e-7) + (1 - labels) * torch.log(1 - output + 1e-7)
        return -torch.mean(loss)


def get_loss(loss: str):
    if loss == Loss.L2.value:
        return L2Loss()
    elif loss == Loss.BCE.value:
        return BCELoss()
    else:
        return None

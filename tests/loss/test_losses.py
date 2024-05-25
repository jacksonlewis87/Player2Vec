import pytest
import torch

from loss.losses import Loss, L2Loss, WeightedL2Loss, BCELoss, get_loss


def test_l2_loss():
    output = torch.tensor(
        [
            [0.0, 1.0],
            [2.0, 3.0],
        ]
    )  # (B, N)
    labels = torch.tensor(
        [
            [1, 1],
            [0, 0],
        ]
    )  # (B, N)
    loss = L2Loss()

    result = loss.forward(output=output, labels=labels)

    assert torch.round(result, decimals=4) == torch.tensor(7)


def test_weighted_l2_loss():
    output = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 4.0, 5.0],
        ]
    )  # (B, N)
    labels = torch.tensor(
        [
            [1, 1, 1],
            [0, 0, 0],
        ]
    )  # (B, N)
    weights = torch.tensor([0.0, 2.0, 0.0])  # softmax -> tensor([0.1065, 0.7870, 0.1065])
    loss = WeightedL2Loss()

    result = loss.forward(output=output, labels=labels, weights=weights)

    assert torch.round(result, decimals=4) == torch.tensor(7.7870)


def test_bce_loss():
    output = torch.tensor([0, 0.5, 0.75, 1])  # (B)
    labels = torch.tensor([0, 0, 1, 1])  # (B)
    loss = BCELoss()

    result = loss.forward(output=output, labels=labels)

    assert torch.round(result, decimals=4) == torch.tensor(0.2452)


@pytest.mark.parametrize(
    "loss, expected_class",
    [(Loss.L2.value, L2Loss), (Loss.BCE.value, BCELoss), (Loss.WEIGHTED_L2.value, WeightedL2Loss), ("test", None)],
)
def test_get_loss(loss, expected_class):
    result = get_loss(loss=loss)

    if expected_class:
        assert isinstance(result, expected_class)
    else:
        assert result is None

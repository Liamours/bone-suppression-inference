from pathlib import Path

import h5py
import torch
from torch import nn

FILTERS = 64
BLOCKS = 16
SCALING = 0.1


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv2d(FILTERS, FILTERS, 3, padding=1)
        self.second = nn.Conv2d(FILTERS, FILTERS, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + SCALING * self.second(torch.relu(self.first(x)))


class ResNetBS(nn.Module):
    """ResNet-BS of Rajaraman et al. 2021: 16 residual blocks, 64 filters, 256x256 grayscale input."""

    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(1, FILTERS, 3, padding=1)
        self.blocks = nn.Sequential(*(ResidualBlock() for _ in range(BLOCKS)))
        self.body = nn.Conv2d(FILTERS, FILTERS, 3, padding=1)
        self.tail = nn.Conv2d(FILTERS, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        return self.tail(x + self.body(self.blocks(x)))


def load_resnet_bs(path: Path) -> ResNetBS:
    model = ResNetBS()
    convs = [model.head]
    for block in model.blocks:
        convs += [block.first, block.second]
    convs += [model.body, model.tail]
    with h5py.File(path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["layer_names"] if str(n).startswith(("conv2d", "b'conv2d"))]
        for name, conv in zip(names, convs):
            group = f[name][name]
            kernel = torch.from_numpy(group["kernel:0"][()]).permute(3, 2, 0, 1)
            conv.weight.data.copy_(kernel)
            conv.bias.data.copy_(torch.from_numpy(group["bias:0"][()]))
    if len(names) != len(convs):
        raise RuntimeError(f"expected {len(convs)} conv layers, found {len(names)}")
    return model.eval()

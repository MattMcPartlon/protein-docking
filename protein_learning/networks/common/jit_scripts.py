import torch
from torch import Tensor, nn


@torch.jit.script
def fused_gelu(x: Tensor) -> Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


class FusedGELUModule(nn.Module):

    def __init__(self):
        super(FusedGELUModule, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return fused_gelu(x)


@torch.jit.script
def fused_gate_n_mul_sigmoid(g: Tensor, x: Tensor) -> Tensor:
    return torch.sigmoid(g) * x


@torch.jit.script
def fused_gate_n_mul_gelu(g: Tensor, x: Tensor) -> Tensor:
    return (g * 0.5 * (1.0 + torch.erf(g / 1.41421))) * x


@torch.jit.script
def fused_scale_n_bias(x: Tensor, bias: Tensor, scale: float) -> Tensor:
    return x * scale + bias


@torch.jit.script
def fused_scale_bias_softmax(x: Tensor, bias: Tensor, scale: float) -> Tensor:
    return torch.softmax(x * scale + bias, dim=-1)


@torch.jit.script
def fused_scale_softmax(x: Tensor, scale: float) -> Tensor:
    return torch.softmax(x * scale, dim=-1)


@torch.jit.script
def fused_dist_attn(x: Tensor, y: Tensor) -> Tensor:
    return -torch.sum(torch.square(x - y), dim=(-1, -2))

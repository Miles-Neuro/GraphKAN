"""GraphKAN with grouped rational activations and configurable ablations."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Linear(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.lin = nn.Linear(inputs, outputs)
        nn.init.xavier_normal_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return self.lin(x)


class RationalLinear(nn.Module):
    def __init__(self, inputs, outputs, groups, activation="swish"):
        super().__init__()
        # Allocate the projection before the external activation for consistent initialization.
        from kat_rational import KAT_Group

        self.lin = nn.Linear(inputs, outputs)
        self.kan = KAT_Group(num_groups=groups, mode=activation)
        nn.init.normal_(self.lin.weight, std=math.sqrt((1.0 if activation == "identity" else 2.8178205270359653) / inputs))
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return self.lin(self.kan(x.contiguous()))


class Branch(nn.Module):
    def __init__(self, inputs, outputs, groups, activation="swish"):
        super().__init__()
        self.lin = RationalLinear(inputs, outputs, groups, activation)


class GraphBranches(nn.Module):
    def __init__(self, inputs, outputs, groups, order, activation="swish"):
        super().__init__()
        self.gc1 = nn.ModuleList(
            [Branch(inputs, outputs, groups, activation) for _ in range(order + 1)]
        )


def supports(weight, order):
    adjacency = F.relu((weight + weight.T) * 0.5)
    degree = adjacency.sum(-1)
    inverse = degree.clamp_min(torch.finfo(adjacency.dtype).eps).rsqrt()
    normalized = inverse.unsqueeze(-1) * adjacency * inverse.unsqueeze(-2)
    identity = torch.eye(len(weight), device=weight.device, dtype=weight.dtype)
    # Normalized L=I-S, fixed lambda_max=2 => scaled L=-S.
    scaled = -normalized
    result = [identity]
    if order:
        result.append(scaled)
    for _ in range(2, order + 1):
        result.append(2 * scaled @ result[-1] - result[-2])
    return result


class GraphKAN(nn.Module):
    def __init__(self, adjacency, config, classes):
        super().__init__()
        inputs, hidden, width = (
            config["in_features"],
            config["hidden_features"],
            config["head_width"],
        )
        self.order = config["chebyshev_order"]
        self.g_adj_weight = nn.Parameter(
            torch.as_tensor(adjacency, dtype=torch.float32).clone()
        )
        self.gcn = GraphBranches(inputs, hidden, config["groups"], self.order, "identity" if config.get("variant") == "5-Identity" else "swish")
        self.out = nn.Sequential(
            nn.Dropout(config["dropout"]),
            Linear(len(adjacency) * hidden, width),
            nn.ReLU(),
            nn.BatchNorm1d(width),
            Linear(width, classes),
        )

        variant=config.get('variant','5-Swish')
        self.variant=variant
        if variant=='w/o Dyn-Adj':self.g_adj_weight.requires_grad_(False)
        if variant=='5-B-spline':
            from .activations import GroupBSpline
            with torch.random.fork_rng(devices=[]):
                for branch in self.gcn.gc1:branch.lin.kan=GroupBSpline()
        elif variant=='w/o KAN':
            for branch in self.gcn.gc1:
                branch.lin=nn.Sequential(nn.Linear(inputs,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        elif variant=='w/o group':
            from .activations import EdgeRationalLinear
            for branch in self.gcn.gc1:branch.lin=EdgeRationalLinear(branch.lin)

    def forward(self, x):
        filtered = [
            branch.lin((support @ x).contiguous())
            for branch, support in zip(
                self.gcn.gc1, supports(self.g_adj_weight, self.order)
            )
        ]
        return self.out(torch.stack(filtered).sum(0).flatten(1))

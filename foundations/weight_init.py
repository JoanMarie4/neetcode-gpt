import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        sd = math.sqrt(2 / (fan_in + fan_out))
        weights = torch.round(torch.randn(fan_out, fan_in) * sd, decimals=4)
        return weights.tolist()


    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        sd = math.sqrt(2/fan_in)
        weights = torch.round(torch.randn(fan_out, fan_in) * sd, decimals=4)
        return weights.tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)

        weights = []
        dim_in = input_dim
        for i in range(num_layers):
            if init_type == "xavier":
                sd = math.sqrt(2 / (dim_in + hidden_dim))
            elif init_type == "kaiming":
                sd = math.sqrt(2 / dim_in)
            else:
                sd = 1.0
            w = torch.randn(hidden_dim, dim_in) * sd
            weights.append(w)
            dim_in = hidden_dim

        sds = []
        x = torch.randn(1, input_dim)
        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            sds.append(round(x.std().item(), 2))

        return sds

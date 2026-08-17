import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        with torch.no_grad():
            dead_fracs = []
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.ReLU):
                    if x.dim() >= 2:
                        dead_frac = round(((x<=0).all(dim=0)).float().mean().item(), 4)
                    else:
                        dead_frac = round((x<=0).float().mean().item(), 4)
                    dead_fracs.append(dead_frac)
        return dead_fracs

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        if len(dead_fractions) == 0:
            return 'healthy'

        max_frac = max(dead_fractions)

        if(max_frac < 0.1):
            return 'healthy'

        if(max_frac > 0.5):
            return 'use_leaky_relu'

        if(dead_fractions[0] > 0.3):
            return 'reinitialize'

        if dead_fractions[len(dead_fractions) - 1] > 0:
            last_frac = 0
            for frac in dead_fractions:
                if frac < last_frac:
                    return 'healthy'
                last_frac = frac
            return 'reduce_learning_rate'
        
        return 'healthy'

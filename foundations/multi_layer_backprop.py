import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        result = {}

        #forward pass
        z1 = np.dot(W1, x) + b1
        a1 = np.maximum(z1, 0)
        z2 = np.dot(W2,a1) + b2
        result['loss'] = np.round(np.mean((z2 - y_true) ** 2), 4)

        #backward pass
        dpred = 2 * (z2 - y_true) / len(y_true)
        
        result['dW2'] = np.round(np.outer(dpred, a1),4)
        result['db2'] = np.round(dpred, 4)
        
        da1 = np.dot(np.array(W2).T, dpred)
        dz1 = da1 * (z1 > 0)

        result['dW1'] = np.round(np.outer(dz1, x),4)
        result['db1'] = np.round(dz1, 4)

        return result

class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        x_old = init
        for i in range(iterations):
            x_new = x_old - learning_rate * 2 * x_old
            x_old = x_new

        result = round(x_old, 5)
        return result

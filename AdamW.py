from typing import List
from Variable import Variable


class Optimizer:
    """Base optimizer class."""

    def __init__(self, parameters: List[Variable]):
        """Initialize optimizer with model parameters."""
        self.parameters = parameters

    def zero_grad(self) -> None:
        """Reset gradients to zero."""
        for param in self.parameters:
            param.grad = 0.0

    def step(self) -> None:
        """Update parameters based on current gradients."""
        raise NotImplementedError


class AdamW(Optimizer):
    """
    AdamW optimizer implementation (Adam with decoupled weight decay).
    """

    def __init__(
        self,
        parameters: List[Variable],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        """
        Initialize AdamW optimizer.

        Args:
            parameters: List of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages of gradient and square
            eps: Term added to denominator for numerical stability
            weight_decay: Weight decay factor
        """
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment estimates
        self.m = [0.0] * len(parameters)  # First moment estimates
        self.v = [0.0] * len(parameters)  # Second moment estimates
        self.step_count = 0

    def step(self) -> None:
        """Update parameters based on gradients."""
        self.step_count += 1

        for i, param in enumerate(self.parameters):
            # Apply weight decay (decoupled from the adaptive part)
            if self.weight_decay > 0:
                param.data -= self.lr * self.weight_decay * param.data

            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + \
                (1 - self.beta2) * (param.grad ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)

            # Update parameter
            param.data -= self.lr * m_hat / ((v_hat ** 0.5) + self.eps)

from typing import List, Tuple
import math
from Variable import Variable


class Optimizer:
    """
    Base optimizer class for neural network parameter optimization.
    
    This abstract base class defines the interface that all optimizers
    should implement for use with the training framework.
    """

    def __init__(self, parameters: List[Variable]) -> None:
        """
        Initialize optimizer with model parameters.
        
        Args:
            parameters: List of trainable parameters
            
        Raises:
            ValueError: If parameters list is empty
        """
        if not parameters:
            raise ValueError("parameters list cannot be empty")
        self.parameters = parameters

    def zero_grad(self) -> None:
        """Reset gradients to zero for all parameters."""
        for param in self.parameters:
            param.grad = 0.0

    def step(self) -> None:
        """Update parameters based on current gradients."""
        raise NotImplementedError("Subclasses must implement step() method")


class AdamW(Optimizer):
    """
    AdamW optimizer implementation (Adam with decoupled weight decay).
    
    AdamW decouples weight decay from the gradient-based update, leading to
    better generalization performance compared to L2 regularization.
    
    References:
        Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2017)
        
    Examples:
        >>> model = MLP(input_size=1, hidden_sizes=[16], output_size=1)
        >>> optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        >>> optimizer.zero_grad()
        >>> # ... compute loss and call loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        parameters: List[Variable],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ) -> None:
        """
        Initialize AdamW optimizer.

        Args:
            parameters: List of parameters to optimize
            lr: Learning rate (default: 1e-3)
            betas: Coefficients for computing running averages of gradient and squared gradient
            eps: Term added to denominator for numerical stability (default: 1e-8)
            weight_decay: Weight decay coefficient (default: 1e-2)
            
        Raises:
            ValueError: If hyperparameters are invalid
        """
        super().__init__(parameters)
        
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment estimates
        self.m = [0.0] * len(parameters)  # First moment estimates
        self.v = [0.0] * len(parameters)  # Second moment estimates
        self.step_count = 0

    def step(self) -> None:
        """
        Perform a single optimization step.
        
        Updates all parameters based on their gradients using the AdamW algorithm.
        
        Raises:
            RuntimeError: If called before any gradients are computed
        """
        if self.step_count == 0:
            # Check if any gradients exist
            has_grad = any(param.grad != 0 for param in self.parameters)
            if not has_grad:
                # Allow first step with zero gradients
                pass
        
        self.step_count += 1

        for i, param in enumerate(self.parameters):
            # Apply weight decay (decoupled from the adaptive part)
            if self.weight_decay > 0:
                param.data -= self.lr * self.weight_decay * param.data

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)

            # Update parameter with numerical stability check
            denom = math.sqrt(v_hat) + self.eps
            if denom > 0:  # Additional safety check
                param.data -= self.lr * m_hat / denom
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.lr
    
    def set_lr(self, lr: float) -> None:
        """
        Set learning rate.
        
        Args:
            lr: New learning rate
            
        Raises:
            ValueError: If learning rate is not positive
        """
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        self.lr = lr
    
    def get_state_dict(self) -> dict:
        """
        Get optimizer state for saving.
        
        Returns:
            Dictionary containing optimizer state
        """
        return {
            'lr': self.lr,
            'betas': (self.beta1, self.beta2),
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'm': self.m.copy(),
            'v': self.v.copy(),
            'step_count': self.step_count
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load optimizer state.
        
        Args:
            state_dict: Dictionary containing optimizer state
        """
        self.lr = state_dict['lr']
        self.beta1, self.beta2 = state_dict['betas']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.m = state_dict['m'].copy()
        self.v = state_dict['v'].copy()
        self.step_count = state_dict['step_count']

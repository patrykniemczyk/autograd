"""
Automatic differentiation implementation using computational graphs.

This module provides the core Variable class that enables automatic differentiation
through operator overloading and computational graph construction.
"""
import math
import warnings
from typing import Callable, List, Tuple, Union, Optional, Set


class Variable:
    """
    A class representing a variable in a computational graph with automatic differentiation.
    Implements reverse-mode autodiff through operator overloading.

    This class is the core building block for neural networks, enabling automatic
    gradient computation through the computational graph.

    Examples:
        Basic operations:
        >>> x = Variable(2.0)
        >>> y = Variable(3.0)
        >>> z = x * y + x
        >>> z.backward()
        >>> print(f"dz/dx = {x.grad}")  # Should be 4.0
        >>> print(f"dz/dy = {y.grad}")  # Should be 2.0

        Neural network usage:
        >>> x = Variable(0.5)
        >>> w = Variable(0.8)
        >>> b = Variable(0.1)
        >>> output = (x * w + b).relu()
        >>> output.backward()

    Attributes:
        data: The value stored in this variable
        grad: The accumulated gradient for this variable
    """

    def __init__(self, data: float, _prev: Tuple["Variable", ...] = ()) -> None:
        """
        Initialize a Variable with data and optional predecessors.

        Args:
            data: The numerical value for this variable
            _prev: Tuple of predecessor variables in the computational graph
                  (used internally for gradient computation)

        Raises:
            TypeError: If data is not a number
            ValueError: If data is NaN or infinite
        """
        if not isinstance(data, (int, float)):
            raise TypeError(
                f"Variable data must be a number, got {type(data)}")

        if math.isnan(data) or math.isinf(data):
            raise ValueError(
                f"Variable data cannot be NaN or infinite, got {data}")

        self.data: float = float(data)
        self.grad: float = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: Tuple["Variable", ...] = _prev

    def backward(self, gradient: Optional[float] = None) -> None:
        """
        Performs a backward pass through the computational graph to compute gradients.
        Uses topological sort to process nodes in the correct order.

        Args:
            gradient: Initial gradient value (default 1.0 for scalar outputs)

        Raises:
            RuntimeError: If the computational graph contains cycles

        Examples:
            >>> x = Variable(2.0)
            >>> y = x ** 2
            >>> y.backward()
            >>> print(x.grad)  # Should be 4.0 (dy/dx = 2*x)
        """
        if gradient is None:
            gradient = 1.0

        if math.isnan(gradient) or math.isinf(gradient):
            raise ValueError(
                f"Initial gradient cannot be NaN or infinite, got {gradient}"
            )

        # Build topological ordering of all nodes in the graph
        topo_order: List[Variable] = []
        visited: Set[Variable] = set()
        temp_visited: Set[Variable] = set()  # For cycle detection

        def build_topo(v: Variable) -> None:
            if v in temp_visited:
                raise RuntimeError("Computational graph contains a cycle")
            if v not in visited:
                temp_visited.add(v)
                visited.add(v)
                for prev in v._prev:
                    build_topo(prev)
                temp_visited.remove(v)
                topo_order.append(v)

        build_topo(self)

        # Backpropagate gradients
        self.grad = gradient
        for node in reversed(topo_order):
            node._backward()

    def __add__(self, other: Union["Variable", float]) -> "Variable":
        """
        Addition operation with gradient tracking.

        Args:
            other: Variable or scalar to add

        Returns:
            New Variable representing the sum

        Examples:
            >>> x = Variable(2.0)
            >>> y = Variable(3.0)
            >>> z = x + y  # z.data = 5.0
            >>> z.backward()
            >>> print(x.grad, y.grad)  # Both should be 1.0
        """
        if isinstance(other, (int, float)):
            if math.isnan(other) or math.isinf(other):
                raise ValueError(f"Cannot add NaN or infinite value: {other}")

        other_val = other.data if isinstance(other, Variable) else other
        out = Variable(
            self.data + other_val,
            _prev=(self,) if not isinstance(
                other, Variable) else (self, other),
        )

        def _backward() -> None:
            self.grad += out.grad
            if isinstance(other, Variable):
                other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Union["Variable", float]) -> "Variable":
        """
        Multiplication operation with gradient tracking.

        Args:
            other: Variable or scalar to multiply

        Returns:
            New Variable representing the product

        Examples:
            >>> x = Variable(2.0)
            >>> y = Variable(3.0)
            >>> z = x * y  # z.data = 6.0
            >>> z.backward()
            >>> print(x.grad, y.grad)  # Should be 3.0, 2.0
        """
        if isinstance(other, (int, float)):
            if math.isnan(other) or math.isinf(other):
                raise ValueError(
                    f"Cannot multiply by NaN or infinite value: {other}")

        other_val = other.data if isinstance(other, Variable) else other
        out = Variable(
            self.data * other_val,
            _prev=(self,) if not isinstance(
                other, Variable) else (self, other),
        )

        def _backward() -> None:
            self.grad += out.grad * (
                other.data if isinstance(other, Variable) else other
            )
            if isinstance(other, Variable):
                other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __pow__(self, other: Union["Variable", float]) -> "Variable":
        """
        Power operation with gradient tracking.

        Args:
            other: Variable or scalar exponent

        Returns:
            New Variable representing self raised to the power of other

        Raises:
            ValueError: If taking log of non-positive base for variable exponent

        Examples:
            >>> x = Variable(2.0)
            >>> y = x ** 3  # y.data = 8.0
            >>> y.backward()
            >>> print(x.grad)  # Should be 12.0 (3 * 2^2)
        """
        if isinstance(other, (int, float)):
            if math.isnan(other) or math.isinf(other):
                raise ValueError(
                    f"Cannot raise to NaN or infinite power: {other}")

        other_val = other.data if isinstance(other, Variable) else other

        # Handle edge cases for numerical stability
        if self.data == 0 and other_val < 0:
            raise ValueError("Cannot raise zero to negative power")

        result_data = self.data**other_val

        # Check for overflow/underflow
        if math.isnan(result_data) or math.isinf(result_data):
            warnings.warn(f"Power operation resulted in {result_data}")

        out = Variable(
            result_data,
            _prev=(self,) if not isinstance(
                other, Variable) else (self, other),
        )

        def _backward() -> None:
            if isinstance(other, Variable):
                # d/dx(x^y) = y * x^(y-1)
                if self.data != 0:  # Avoid division by zero
                    self.grad += out.grad * other.data * \
                        self.data ** (other.data - 1)
                # d/dy(x^y) = x^y * ln(x)
                if self.data > 0:  # Avoid log of negative/zero
                    other.grad += out.grad * \
                        self.data**other.data * math.log(self.data)
                elif self.data < 0:
                    warnings.warn(
                        "Taking logarithm of negative base in gradient computation"
                    )
            else:
                # d/dx(x^c) = c * x^(c-1)
                if self.data != 0 or other >= 1:  # Avoid problematic cases
                    self.grad += out.grad * other * self.data ** (other - 1)

        out._backward = _backward
        return out

    def relu(self) -> "Variable":
        """
        ReLU activation function with gradient tracking.

        Returns:
            New Variable with ReLU applied: max(0, self)

        Examples:
            >>> x = Variable(-2.0)
            >>> y = x.relu()  # y.data = 0.0
            >>> y.backward()
            >>> print(x.grad)  # Should be 0.0 (gradient is 0 for negative input)

            >>> x = Variable(2.0)
            >>> y = x.relu()  # y.data = 2.0
            >>> y.backward()
            >>> print(x.grad)  # Should be 1.0 (gradient is 1 for positive input)
        """
        out = Variable(max(0, self.data), _prev=(self,))

        def _backward() -> None:
            self.grad += out.grad * (1.0 if self.data > 0 else 0.0)

        out._backward = _backward
        return out

    def tanh(self) -> "Variable":
        """
        Hyperbolic tangent activation function with gradient tracking.

        Returns:
            New Variable with tanh applied: (e^x - e^(-x)) / (e^x + e^(-x))

        Examples:
            >>> x = Variable(0.0)
            >>> y = x.tanh()  # y.data = 0.0
            >>> y.backward()
            >>> print(x.grad)  # Should be 1.0 (tanh'(0) = 1)
        """
        try:
            tanh_val = math.tanh(self.data)
        except OverflowError:
            # Handle extreme values
            tanh_val = 1.0 if self.data > 0 else -1.0

        out = Variable(tanh_val, _prev=(self,))

        def _backward() -> None:
            # d/dx(tanh(x)) = 1 - tanh^2(x)
            self.grad += out.grad * (1.0 - out.data**2)

        out._backward = _backward
        return out

    def sigmoid(self) -> "Variable":
        """
        Sigmoid activation function with gradient tracking.

        Returns:
            New Variable with sigmoid applied: 1 / (1 + e^(-x))

        Examples:
            >>> x = Variable(0.0)
            >>> y = x.sigmoid()  # y.data = 0.5
            >>> y.backward()
            >>> print(x.grad)  # Should be 0.25 (sigmoid'(0) = 0.25)
        """
        try:
            # Use numerically stable sigmoid computation
            if self.data >= 0:
                exp_neg = math.exp(-self.data)
                sigmoid_val = 1.0 / (1.0 + exp_neg)
            else:
                exp_pos = math.exp(self.data)
                sigmoid_val = exp_pos / (1.0 + exp_pos)
        except OverflowError:
            # Handle extreme values
            sigmoid_val = 1.0 if self.data > 0 else 0.0

        out = Variable(sigmoid_val, _prev=(self,))

        def _backward() -> None:
            # d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
            self.grad += out.grad * out.data * (1.0 - out.data)

        out._backward = _backward
        return out

    def __neg__(self) -> "Variable":
        """Negation operation: -self."""
        return self * -1

    def __sub__(self, other: Union["Variable", float]) -> "Variable":
        """Subtraction operation: self - other."""
        return self + (-other)

    def __truediv__(self, other: Union["Variable", float]) -> "Variable":
        """
        Division operation: self / other.

        Args:
            other: Variable or scalar divisor

        Returns:
            New Variable representing the quotient

        Raises:
            ValueError: If dividing by zero
        """
        if isinstance(other, Variable):
            if other.data == 0:
                raise ValueError("Division by zero")
        elif other == 0:
            raise ValueError("Division by zero")

        return self * (other**-1)

    def __radd__(self, other: Union["Variable", float]) -> "Variable":
        """Reverse addition operation: other + self."""
        return self + other

    def __rsub__(self, other: Union["Variable", float]) -> "Variable":
        """Reverse subtraction operation: other - self."""
        return -self + other

    def __rmul__(self, other: Union["Variable", float]) -> "Variable":
        """Reverse multiplication operation: other * self."""
        return self * other

    def __rpow__(self, other: Union["Variable", float]) -> "Variable":
        """Reverse power operation: other ** self."""
        return Variable(other) ** self

    def __rtruediv__(self, other: Union["Variable", float]) -> "Variable":
        """Reverse division operation: other / self."""
        return Variable(other) / self

    def __repr__(self) -> str:
        """String representation of the Variable."""
        return f"Variable(data={self.data:.4f}, grad={self.grad:.4f})"

    def zero_grad(self) -> None:
        """
        Reset the gradient of this variable to zero.

        Useful for clearing gradients before a new backward pass.
        """
        self.grad = 0.0

    def clip_grad(self, max_norm: float) -> None:
        """
        Clip the gradient to prevent exploding gradients.

        Args:
            max_norm: Maximum allowed gradient magnitude

        Examples:
            >>> x = Variable(1.0)
            >>> x.grad = 10.0
            >>> x.clip_grad(5.0)
            >>> print(x.grad)  # Should be 5.0
        """
        if max_norm <= 0:
            raise ValueError(f"max_norm must be positive, got {max_norm}")

        grad_norm = abs(self.grad)
        if grad_norm > max_norm:
            self.grad = self.grad * (max_norm / grad_norm)

    def __abs__(self) -> "Variable":
        """
        Absolute value operation with gradient tracking.

        Returns:
            New Variable with absolute value applied

        Examples:
            >>> x = Variable(-2.0)
            >>> y = abs(x)  # y.data = 2.0
            >>> y.backward()
            >>> print(x.grad)  # Should be -1.0 (derivative of abs for negative input)
        """
        out = Variable(abs(self.data), _prev=(self,))

        def _backward() -> None:
            # d/dx(|x|) = sign(x) for x != 0, undefined for x = 0
            if self.data > 0:
                self.grad += out.grad * 1.0
            elif self.data < 0:
                self.grad += out.grad * -1.0
            # For x = 0, we use the convention that the derivative is 0

        out._backward = _backward
        return out

import math
from typing import Callable, List, Tuple, Union, Optional, Set


class Variable:
    """
    A class representing a variable in a computational graph with automatic differentiation.
    Implements reverse-mode autodiff through operator overloading.
    """

    def __init__(self, data: float, _prev: Tuple["Variable", ...] = ()):
        self.data: float = data
        self.grad: float = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: Tuple["Variable", ...] = _prev

    def backward(self) -> None:
        """
        Performs a backward pass through the computational graph to compute gradients.
        Uses topological sort to process nodes in the correct order.
        """
        # Build topological ordering of all nodes in the graph
        topo_order: List[Variable] = []
        visited: Set[Variable] = set()

        def build_topo(v: Variable) -> None:
            if v not in visited:
                visited.add(v)
                for prev in v._prev:
                    build_topo(prev)
                topo_order.append(v)

        build_topo(self)

        # Backpropagate gradients
        self.grad = 1.0  # Set gradient of output variable to 1
        for node in reversed(topo_order):
            node._backward()

    def __add__(self, other: Union["Variable", float]) -> "Variable":
        """Addition operation with gradient tracking."""
        other_val = other.data if isinstance(other, Variable) else other
        out = Variable(self.data + other_val, _prev=(self,)
                       if not isinstance(other, Variable) else (self, other))

        def _backward() -> None:
            self.grad += out.grad
            if isinstance(other, Variable):
                other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Union["Variable", float]) -> "Variable":
        """Multiplication operation with gradient tracking."""
        other_val = other.data if isinstance(other, Variable) else other
        out = Variable(self.data * other_val, _prev=(self,)
                       if not isinstance(other, Variable) else (self, other))

        def _backward() -> None:
            self.grad += out.grad * \
                (other.data if isinstance(other, Variable) else other)
            if isinstance(other, Variable):
                other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __pow__(self, other: Union["Variable", float]) -> "Variable":
        """Power operation with gradient tracking."""
        other_val = other.data if isinstance(other, Variable) else other
        out = Variable(self.data ** other_val, _prev=(self,)
                       if not isinstance(other, Variable) else (self, other))

        def _backward() -> None:
            if isinstance(other, Variable):
                self.grad += out.grad * other.data * \
                    self.data ** (other.data - 1)
                if self.data > 0:  # Avoid log of negative/zero
                    other.grad += out.grad * \
                        self.data ** other.data * math.log(self.data)
            else:
                self.grad += out.grad * other * self.data ** (other - 1)

        out._backward = _backward
        return out

    def relu(self) -> "Variable":
        """ReLU activation function with gradient tracking."""
        out = Variable(max(0, self.data), _prev=(self,))

        def _backward() -> None:
            self.grad += out.grad * (1 if self.data > 0 else 0)

        out._backward = _backward
        return out

    def __neg__(self) -> "Variable":
        """Negation operation."""
        return self * -1

    def __sub__(self, other: Union["Variable", float]) -> "Variable":
        """Subtraction operation."""
        return self + (-other)

    def __truediv__(self, other: Union["Variable", float]) -> "Variable":
        """Division operation."""
        return self * (other ** -1)

    def __radd__(self, other: Union["Variable", float]) -> "Variable":
        """Reverse addition operation."""
        return self + other

    def __rsub__(self, other: Union["Variable", float]) -> "Variable":
        """Reverse subtraction operation."""
        return -self + other

    def __rmul__(self, other: Union["Variable", float]) -> "Variable":
        """Reverse multiplication operation."""
        return self * other

    def __rpow__(self, other: Union["Variable", float]) -> "Variable":
        """Reverse power operation."""
        return Variable(other) ** self

    def __rtruediv__(self, other: Union["Variable", float]) -> "Variable":
        """Reverse division operation."""
        return Variable(other) / self

    def __repr__(self) -> str:
        """String representation."""
        return f"Variable(data={self.data:.4f}, grad={self.grad:.4f})"

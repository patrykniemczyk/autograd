import math
import random
from typing import List
from Variable import Variable


class Neuron():
    """Single neuron with weights, bias, and activation."""

    def __init__(self, input_size: int, initialization_gain: float = 2.0):
        """
        Initialize a neuron with random weights and zero bias.

        Args:
            input_size: Number of input features
            initialization_gain: Gain factor for He initialization
        """
        # He initialization
        std = math.sqrt(initialization_gain / input_size)
        self.weights = [Variable(random.gauss(0, std))
                        for _ in range(input_size)]
        self.bias = Variable(0.0)

    def __call__(self, inputs: List[Variable]) -> Variable:
        """Compute the neuron's output for given inputs."""
        if len(inputs) != len(self.weights):
            raise ValueError(
                f"Expected {len(self.weights)} inputs, got {len(inputs)}")

        # Weighted sum + bias
        return sum(w * x for w, x in zip(self.weights, inputs)) + self.bias

    def parameters(self) -> List[Variable]:
        """Return all trainable parameters."""
        return self.weights + [self.bias]


class Linear():
    """Linear (fully connected) layer."""

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize a linear layer.

        Args:
            input_size: Number of input features
            output_size: Number of output features
        """
        self.neurons = [Neuron(input_size) for _ in range(output_size)]
        self.input_size = input_size
        self.output_size = output_size

    def __call__(self, inputs: List[Variable]) -> List[Variable]:
        """Forward pass of linear layer."""
        if len(inputs) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} inputs, got {len(inputs)}")

        return [neuron(inputs) for neuron in self.neurons]

    def parameters(self) -> List[Variable]:
        """Return all trainable parameters."""
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class ReLU():
    """ReLU activation function module."""

    def __call__(self, inputs: List[Variable]) -> List[Variable]:
        """Apply ReLU activation to each input."""
        return [x.relu() for x in inputs]
    
    def parameters(self) -> List[Variable]:
        """Return all trainable parameters."""
        return []


class MLP():
    """Multi-layer perceptron neural network."""

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        """
        Initialize an MLP with specified architecture.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output features
        """
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = []

        # Create layers
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i+1]))
            # Add ReLU after all but the last layer
            if i < len(sizes) - 2:
                self.layers.append(ReLU())

    def __call__(self, inputs: List[Variable]) -> List[Variable]:
        """Forward pass through the entire network."""
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Variable]:
        """Return all trainable parameters."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

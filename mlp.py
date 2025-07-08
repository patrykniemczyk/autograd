import math
import random
from typing import List, Optional, Union, Literal
from variable import Variable


class Neuron:
    """
    Single neuron with weights, bias, and activation.

    This class represents a single neuron in a neural network, implementing
    the basic linear transformation: output = sum(weights * inputs) + bias

    Examples:
        >>> neuron = Neuron(input_size=3)
        >>> inputs = [Variable(1.0), Variable(2.0), Variable(3.0)]
        >>> output = neuron(inputs)
    """

    def __init__(self, input_size: int, initialization_gain: float = 2.0) -> None:
        """
        Initialize a neuron with random weights and zero bias.

        Args:
            input_size: Number of input features
            initialization_gain: Gain factor for He initialization

        Raises:
            ValueError: If input_size is not positive
        """
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if initialization_gain <= 0:
            raise ValueError(
                f"initialization_gain must be positive, got {initialization_gain}"
            )

        # He initialization for ReLU networks
        std = math.sqrt(initialization_gain / input_size)
        self.weights = [Variable(random.gauss(0, std)) for _ in range(input_size)]
        self.bias = Variable(0.0)
        self.input_size = input_size

    def __call__(self, inputs: List[Variable]) -> Variable:
        """
        Compute the neuron's output for given inputs.

        Args:
            inputs: List of input Variables

        Returns:
            Output Variable representing weighted sum + bias

        Raises:
            ValueError: If number of inputs doesn't match expected size
        """
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")

        # Weighted sum + bias
        return sum(w * x for w, x in zip(self.weights, inputs)) + self.bias

    def parameters(self) -> List[Variable]:
        """Return all trainable parameters (weights + bias)."""
        return self.weights + [self.bias]


class Linear:
    """
    Linear (fully connected) layer.

    Applies a linear transformation to the incoming data: y = xW^T + b
    where W is the weight matrix and b is the bias vector.

    Examples:
        >>> layer = Linear(input_size=3, output_size=2)
        >>> inputs = [Variable(1.0), Variable(2.0), Variable(3.0)]
        >>> outputs = layer(inputs)
        >>> print(len(outputs))  # Should be 2
    """

    def __init__(
        self, input_size: int, output_size: int, initialization_gain: float = 2.0
    ) -> None:
        """
        Initialize a linear layer.

        Args:
            input_size: Number of input features
            output_size: Number of output features
            initialization_gain: Gain factor for weight initialization

        Raises:
            ValueError: If input_size or output_size is not positive
        """
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")

        self.neurons = [
            Neuron(input_size, initialization_gain) for _ in range(output_size)
        ]
        self.input_size = input_size
        self.output_size = output_size

    def __call__(self, inputs: List[Variable]) -> List[Variable]:
        """
        Forward pass of linear layer.

        Args:
            inputs: List of input Variables

        Returns:
            List of output Variables

        Raises:
            ValueError: If number of inputs doesn't match expected size
        """
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")

        return [neuron(inputs) for neuron in self.neurons]

    def parameters(self) -> List[Variable]:
        """Return all trainable parameters from all neurons."""
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class ReLU:
    """
    ReLU activation function module.

    Applies the ReLU function element-wise: ReLU(x) = max(0, x)
    """

    def __call__(self, inputs: List[Variable]) -> List[Variable]:
        """Apply ReLU activation to each input."""
        return [x.relu() for x in inputs]

    def parameters(self) -> List[Variable]:
        """Return all trainable parameters (none for activation functions)."""
        return []


class Tanh:
    """
    Hyperbolic tangent activation function module.

    Applies the tanh function element-wise: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    """

    def __call__(self, inputs: List[Variable]) -> List[Variable]:
        """Apply tanh activation to each input."""
        return [x.tanh() for x in inputs]

    def parameters(self) -> List[Variable]:
        """Return all trainable parameters (none for activation functions)."""
        return []


class Sigmoid:
    """
    Sigmoid activation function module.

    Applies the sigmoid function element-wise: sigmoid(x) = 1 / (1 + e^(-x))
    """

    def __call__(self, inputs: List[Variable]) -> List[Variable]:
        """Apply sigmoid activation to each input."""
        return [x.sigmoid() for x in inputs]

    def parameters(self) -> List[Variable]:
        """Return all trainable parameters (none for activation functions)."""
        return []


# Type alias for activation functions
ActivationType = Literal["relu", "tanh", "sigmoid"]


def get_activation(activation: ActivationType) -> Union[ReLU, Tanh, Sigmoid]:
    """
    Factory function to get activation function by name.

    Args:
        activation: Name of the activation function

    Returns:
        Activation function instance

    Raises:
        ValueError: If activation name is not supported
    """
    activations = {"relu": ReLU(), "tanh": Tanh(), "sigmoid": Sigmoid()}

    if activation not in activations:
        raise ValueError(
            f"Unsupported activation: {activation}. "
            f"Supported: {list(activations.keys())}"
        )

    return activations[activation]


class MLP:
    """
    Multi-layer perceptron neural network.

    A feedforward neural network with configurable architecture and activation functions.

    Examples:
        Basic usage:
        >>> mlp = MLP(input_size=2, hidden_sizes=[16, 8], output_size=1)
        >>> inputs = [Variable(1.0), Variable(2.0)]
        >>> outputs = mlp(inputs)

        With custom activation:
        >>> mlp = MLP(input_size=2, hidden_sizes=[16], output_size=1,
        ...           activation='tanh')
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: ActivationType = "relu",
    ) -> None:
        """
        Initialize an MLP with specified architecture.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output features
            activation: Activation function to use between layers

        Raises:
            ValueError: If any size parameter is not positive
        """
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        if any(size <= 0 for size in hidden_sizes):
            raise ValueError(f"All hidden sizes must be positive, got {hidden_sizes}")

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes.copy()
        self.output_size = output_size
        self.activation_name = activation

        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = []

        # Create layers
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1]))
            # Add activation after all but the last layer
            if i < len(sizes) - 2:
                self.layers.append(get_activation(activation))

    def __call__(self, inputs: List[Variable]) -> List[Variable]:
        """
        Forward pass through the entire network.

        Args:
            inputs: List of input Variables

        Returns:
            List of output Variables

        Raises:
            ValueError: If number of inputs doesn't match expected size
        """
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")

        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Variable]:
        """Return all trainable parameters from all layers."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return len(self.parameters())

    def __repr__(self) -> str:
        """String representation of the MLP."""
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        arch_str = " -> ".join(map(str, layer_sizes))
        return f"MLP({arch_str}, activation={self.activation_name}, params={self.num_parameters()})"

import math
import random
from autograd import Variable


class Neuron:

    def __init__(self, input_size):
        self.w = [Variable(random.gauss(0, 1) * math.sqrt(2 / input_size))
                  for _ in range(input_size)]
        self.b = Variable(0.0)

    def __call__(self, activations):
        return sum(w * a for w, a in zip(self.w, activations)) + self.b

    def parameters(self):
        return self.w + [self.b]


class Linear:

    def __init__(self, input_size, output_size):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def __call__(self, activations):
        return [neuron(activations) for neuron in self.neurons]

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]


class ReLU:

    def __call__(self, activations):
        return [a.relu() for a in activations]

    def parameters(self):
        return []


class MLP:

    def __init__(self, input_size, hidden_sizes, output_size):
        self.layers = []
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(Linear(prev_size, size))
            self.layers.append(ReLU())
            prev_size = size
        self.layers.append(Linear(prev_size, output_size))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

# Import necessary components
from autograd import Variable
from neural_net import MLP
from adamw import AdamW
from dataset import PolynomialDataset

# Create a polynomial dataset (3x^2 + 2x - 1)
dataset = PolynomialDataset([3, 2, -1], noise_std=0.01)

# Initialize a model (MLP with 1 input, 2 hidden layers of 16 neurons, 1 output)
model = MLP(1, [16, 16], 1)

# Create optimizer
optimizer = AdamW(model.parameters(), lr=0.01)

epochs = 100
# Training loop
for epoch in range(epochs+1):
    # Zero gradients
    optimizer.zero_grad()

    # Forward pass and compute loss
    loss = Variable(0.0)
    for x, y in dataset:
        output = model(x)
        loss += sum((o - t) ** 2 for o, t in zip(output, y))
    loss /= len(dataset)

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss: {loss.data:.2f}")

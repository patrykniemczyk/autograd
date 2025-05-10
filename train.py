from autograd import Variable
from mlp import MLP
from dataset import PolynomialDataset


def mse(output, target):
    loss = Variable(0.0)
    for o, t in zip(output, target):
        loss += (o - t) ** 2
    return loss / len(output)


def lossfn(dataset, model):
    loss = Variable(0.0)
    for x, y in dataset:
        output = model(x)
        loss += mse(output, y)

    loss /= len(dataset)
    return loss


class AdamW:

    def __init__(self, parameters, lr=0.001, weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0] * len(parameters)
        self.v = [0.0] * len(parameters)
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                self.m[i] = self.beta1 * self.m[i] + \
                    (1 - self.beta1) * param.grad
                self.v[i] = self.beta2 * self.v[i] + \
                    (1 - self.beta2) * (param.grad ** 2)
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                param.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
                param.data -= self.weight_decay * param.data

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0.0


dataset = PolynomialDataset([3, 2, -1], noise_std=0.01)

model = MLP(1, [16, 16], 1)
optimizer = AdamW(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = lossfn(dataset, model)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        testloss = lossfn(dataset, model)
        print(f"Epoch {epoch}, test loss: {testloss.data:.2f}")

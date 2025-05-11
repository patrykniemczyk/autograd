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

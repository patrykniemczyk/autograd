import random
from autograd import Variable


class PolynomialDataset:
    def __init__(self, coefficients, num_samples=10000, noise_std=0.1, test_ratio=0.2):
        self.coefficients = [Variable(c) for c in coefficients]
        self.num_samples = num_samples
        self.noise_std = noise_std

        full_x = [Variable(random.uniform(-1, 1)) for _ in range(num_samples)]
        full_y = [self._generate_polynomial(
            xi) + Variable(random.gauss(0, noise_std)) for xi in full_x]

        split_idx = int(num_samples * (1 - test_ratio))
        self.train_x = full_x[:split_idx]
        self.train_y = full_y[:split_idx]
        self.test_x = full_x[split_idx:]
        self.test_y = full_y[split_idx:]

    def _generate_polynomial(self, x):
        degree = len(self.coefficients)
        return sum(self.coefficients[i] * (x ** (degree - i - 1)) for i in range(degree))

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return self.train_x[idx], self.train_y[idx]

    def get_test_set(self):
        return list(zip(self.test_x, self.test_y))


def get_batches(dataset, batch_size, shuffle=True):
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(dataset), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch = [dataset[i] for i in batch_indices]
        yield [b[0] for b in batch], [b[1] for b in batch]

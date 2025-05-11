import random
from typing import List, Tuple
from Variable import Variable


class Dataset:
    """Base dataset class."""

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[List[Variable], List[Variable]]:
        """Get a sample by index."""
        raise NotImplementedError

    def get_batch(self, batch_size: int) -> Tuple[List[List[Variable]], List[List[Variable]]]:
        """Get a random batch of samples."""
        raise NotImplementedError


class PolynomialDataset(Dataset):
    """Dataset for polynomial regression tasks."""

    def __init__(
        self,
        coefficients: List[float],
        domain: Tuple[float, float] = (-1.0, 1.0),
        num_samples: int = 10000,
        noise_std: float = 0.1,
        test_ratio: float = 0.2
    ):
        """
        Initialize polynomial dataset.

        Args:
            coefficients: List of polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]
                          For example, [3, 2, -1] represents 3x^2 + 2x - 1
            domain: Range (min, max) for generating x values
            num_samples: Total number of samples to generate
            noise_std: Standard deviation of Gaussian noise added to outputs
            test_ratio: Fraction of data to use for testing
        """
        self.coefficients = [Variable(c) for c in coefficients]
        self.domain = domain
        self.num_samples = num_samples
        self.noise_std = noise_std

        # Generate full dataset
        full_x = []
        full_y = []

        for _ in range(num_samples):
            # Generate input sample
            x_val = random.uniform(*domain)
            x = [Variable(x_val)]

            # Generate target with noise
            y_val = self._evaluate_polynomial(
                x_val) + random.gauss(0, noise_std)
            y = [Variable(y_val)]

            full_x.append(x)
            full_y.append(y)

        # Split into train and test sets
        split_idx = int(num_samples * (1 - test_ratio))
        self.train_x = full_x[:split_idx]
        self.train_y = full_y[:split_idx]
        self.test_x = full_x[split_idx:]
        self.test_y = full_y[split_idx:]

    def _evaluate_polynomial(self, x: float) -> float:
        """Evaluate the polynomial at point x."""
        result = 0.0
        for i, coef in enumerate(self.coefficients):
            power = len(self.coefficients) - i - 1
            result += coef.data * (x ** power)
        return result

    def __len__(self) -> int:
        """Return the number of training samples."""
        return len(self.train_x)

    def __getitem__(self, idx: int) -> Tuple[List[Variable], List[Variable]]:
        """Get a training sample by index."""
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range")
        return self.train_x[idx], self.train_y[idx]

    def get_batch(self, batch_size: int) -> Tuple[List[List[Variable]], List[List[Variable]]]:
        """Get a random batch of training samples."""
        if batch_size > len(self):
            raise ValueError(
                f"Batch size {batch_size} larger than dataset size {len(self)}")

        indices = random.sample(range(len(self)), batch_size)
        batch_x = [self.train_x[i] for i in indices]
        batch_y = [self.train_y[i] for i in indices]

        return batch_x, batch_y

    def get_test_data(self) -> Tuple[List[List[Variable]], List[List[Variable]]]:
        """Get all test data."""
        return self.test_x, self.test_y

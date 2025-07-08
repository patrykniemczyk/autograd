"""
Dataset utilities for neural network training.

This module provides dataset classes for generating and managing training data,
specifically for polynomial regression and other mathematical functions.
"""
import random
from typing import List, Tuple, Optional
from variable import Variable


class Dataset:
    """
    Base dataset class for neural network training.

    This abstract base class defines the interface that all datasets
    should implement for use with the training framework.
    """

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[List[Variable], List[Variable]]:
        """Get a sample by index."""
        raise NotImplementedError

    def get_batch(
        self, batch_size: int
    ) -> Tuple[List[List[Variable]], List[List[Variable]]]:
        """Get a random batch of samples."""
        raise NotImplementedError


class PolynomialDataset(Dataset):
    """
    Dataset for polynomial regression tasks.

    Generates data based on a polynomial function with optional noise.
    Supports automatic train/test splitting and various polynomial types.

    Examples:
        Simple quadratic:
        >>> dataset = PolynomialDataset([1, 0, -1])  # x^2 - 1
        >>> x, y = dataset[0]

        Cubic with noise:
        >>> dataset = PolynomialDataset([2, -1, 0, 3], noise_std=0.1)
    """

    def __init__(
        self,
        coefficients: List[float],
        domain: Tuple[float, float] = (-1.0, 1.0),
        num_samples: int = 10000,
        noise_std: float = 0.1,
        test_ratio: float = 0.2,
        normalize_inputs: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize polynomial dataset.

        Args:
            coefficients: List of polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]
                          For example, [3, 2, -1] represents 3x^2 + 2x - 1
            domain: Range (min, max) for generating x values
            num_samples: Total number of samples to generate
            noise_std: Standard deviation of Gaussian noise added to outputs
            test_ratio: Fraction of data to use for testing (0.0 to 1.0)
            normalize_inputs: Whether to normalize input values to [-1, 1]
            seed: Random seed for reproducible data generation

        Raises:
            ValueError: If parameters are invalid
        """
        if not coefficients:
            raise ValueError("coefficients cannot be empty")
        if len(domain) != 2 or domain[1] <= domain[0]:
            raise ValueError(
                f"domain must be (min, max) with min < max, got {domain}")
        if num_samples <= 0:
            raise ValueError(
                f"num_samples must be positive, got {num_samples}")
        if noise_std < 0:
            raise ValueError(
                f"noise_std must be non-negative, got {noise_std}")
        if not 0 <= test_ratio <= 1:
            raise ValueError(
                f"test_ratio must be between 0 and 1, got {test_ratio}")

        if seed is not None:
            random.seed(seed)

        self.coefficients = [Variable(c) for c in coefficients]
        self.domain = domain
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.normalize_inputs = normalize_inputs

        # Generate full dataset
        full_x = []
        full_y = []

        for _ in range(num_samples):
            # Generate input sample
            x_val = random.uniform(*domain)

            # Normalize if requested
            if normalize_inputs:
                x_normalized = 2 * \
                    (x_val - domain[0]) / (domain[1] - domain[0]) - 1
                x = [Variable(x_normalized)]
            else:
                x = [Variable(x_val)]

            # Generate target with noise
            y_val = self._evaluate_polynomial(x_val)
            if noise_std > 0:
                y_val += random.gauss(0, noise_std)
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
        """
        Evaluate the polynomial at point x.

        Args:
            x: Input value

        Returns:
            Polynomial value at x
        """
        result = 0.0
        for i, coef in enumerate(self.coefficients):
            power = len(self.coefficients) - i - 1
            result += coef.data * (x**power)
        return result

    def __len__(self) -> int:
        """Return the number of training samples."""
        return len(self.train_x)

    def __getitem__(self, idx: int) -> Tuple[List[Variable], List[Variable]]:
        """
        Get a training sample by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input, target) Variable lists

        Raises:
            IndexError: If index is out of range
        """
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        return self.train_x[idx], self.train_y[idx]

    def get_batch(
        self, batch_size: int
    ) -> Tuple[List[List[Variable]], List[List[Variable]]]:
        """
        Get a random batch of training samples.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Tuple of (batch_inputs, batch_targets)

        Raises:
            ValueError: If batch_size is larger than dataset or invalid
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > len(self):
            raise ValueError(
                f"Batch size {batch_size} larger than dataset size {len(self)}"
            )

        indices = random.sample(range(len(self)), batch_size)
        batch_x = [self.train_x[i] for i in indices]
        batch_y = [self.train_y[i] for i in indices]

        return batch_x, batch_y

    def get_test_data(self) -> Tuple[List[List[Variable]], List[List[Variable]]]:
        """
        Get all test data.

        Returns:
            Tuple of (test_inputs, test_targets)
        """
        return self.test_x, self.test_y

    def get_polynomial_string(self) -> str:
        """
        Get a string representation of the polynomial.

        Returns:
            Human-readable polynomial string
        """
        if not self.coefficients:
            return "0"

        terms = []
        degree = len(self.coefficients) - 1

        for i, coef in enumerate(self.coefficients):
            power = degree - i
            coef_val = coef.data

            if coef_val == 0:
                continue

            # Format coefficient
            if power == 0:
                term = f"{coef_val:g}"
            elif power == 1:
                if coef_val == 1:
                    term = "x"
                elif coef_val == -1:
                    term = "-x"
                else:
                    term = f"{coef_val:g}x"
            else:
                if coef_val == 1:
                    term = f"x^{power}"
                elif coef_val == -1:
                    term = f"-x^{power}"
                else:
                    term = f"{coef_val:g}x^{power}"

            terms.append(term)

        if not terms:
            return "0"

        # Join terms with appropriate signs
        result = terms[0]
        for term in terms[1:]:
            if term.startswith("-"):
                result += f" - {term[1:]}"
            else:
                result += f" + {term}"

        return result

    def __repr__(self) -> str:
        """String representation of the dataset."""
        poly_str = self.get_polynomial_string()
        return (
            f"PolynomialDataset(polynomial='{poly_str}', "
            f"samples={self.num_samples}, train={len(self)}, "
            f"test={len(self.test_x)}, noise_std={self.noise_std})"
        )

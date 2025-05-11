import argparse
import time
from typing import List, Dict

from Variable import Variable
from MLP import MLP
from AdamW import AdamW
from Dataset import PolynomialDataset


def mean_squared_error(predictions: List[Variable], targets: List[Variable]) -> Variable:
    """Compute mean squared error between predictions and targets."""
    if len(predictions) != len(targets):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) doesn't match targets ({len(targets)})")

    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)


class Trainer:
    """Class to handle model training and evaluation."""

    def __init__(
        self,
        model: MLP,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        weight_decay: float = 0.01,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            optimizer_name: Name of optimizer ('adamw' or 'sgd')
            learning_rate: Learning rate for optimization
            batch_size: Number of samples per batch
            weight_decay: Weight decay factor
        """
        self.model = model
        self.batch_size = batch_size

        # Create optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def train_epoch(self, dataset, verbose: bool = False) -> float:
        """
        Train for one epoch.

        Args:
            dataset: Dataset to train on
            verbose: Whether to print batch losses

        Returns:
            Average loss for the epoch
        """
        num_batches = max(1, len(dataset) // self.batch_size)
        epoch_loss = 0.0

        for batch_idx in range(num_batches):
            # Get batch
            batch_x, batch_y = dataset.get_batch(self.batch_size)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            batch_loss = Variable(0.0)
            for x, y in zip(batch_x, batch_y):
                output = self.model(x)
                batch_loss = batch_loss + mean_squared_error(output, y)

            # Average batch loss
            batch_loss = batch_loss / len(batch_x)

            # Backward pass
            batch_loss.backward()

            # Update parameters
            self.optimizer.step()

            # Track loss
            epoch_loss += batch_loss.data

            if verbose and batch_idx % 10 == 0:
                print(
                    f"  Batch {batch_idx}/{num_batches}, loss: {batch_loss.data:.6f}")

        # Return average loss
        return epoch_loss / num_batches

    def evaluate(self, dataset) -> float:
        """
        Evaluate model on dataset.

        Args:
            dataset: Dataset to evaluate on

        Returns:
            Average loss
        """
        if hasattr(dataset, "get_test_data"):
            # Use test data if available
            test_x, test_y = dataset.get_test_data()
        else:
            # Use all data
            test_x, test_y = [], []
            for i in range(len(dataset)):
                x, y = dataset[i]
                test_x.append(x)
                test_y.append(y)

        total_loss = 0.0
        for x, y in zip(test_x, test_y):
            output = self.model(x)
            loss = mean_squared_error(output, y)
            total_loss += loss.data

        return total_loss / len(test_x)

    def train(
        self,
        dataset,
        epochs: int = 100,
        validate_every: int = 1,
        verbose: bool = True
    ) -> List[Dict[str, float]]:
        """
        Train the model.

        Args:
            dataset: Dataset to train on
            epochs: Number of epochs to train
            validate_every: Validate every N epochs
            verbose: Whether to print progress

        Returns:
            List of training metrics per epoch
        """
        metrics = []

        start_time = time.time()
        for epoch in range(epochs + 1):
            epoch_metrics = {}

            # Train one epoch (skip for epoch 0)
            if epoch > 0:
                train_loss = self.train_epoch(dataset, verbose=False)
                epoch_metrics["train_loss"] = train_loss

            # Validate
            if epoch % validate_every == 0 or epoch == epochs:
                eval_loss = self.evaluate(dataset)
                epoch_metrics["eval_loss"] = eval_loss

                if verbose:
                    elapsed = time.time() - start_time
                    metrics_str = ", ".join(
                        [f"{k}: {v:.6f}" for k, v in epoch_metrics.items()])
                    print(
                        f"Epoch {epoch}/{epochs}, {metrics_str}, time: {elapsed:.2f}s")

            metrics.append(epoch_metrics)

        return metrics


def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description="Train an MLP model")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[4, 4],
                        help="Sizes of hidden layers")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay factor")
    parser.add_argument("--noise", type=float, default=0.01,
                        help="Noise level for dataset")
    parser.add_argument("--polynomial-coeffs", type=float, nargs="+", default=[3, 1, -1],
                        help="Coefficients of the polynomial, from highest to lowest power")

    args = parser.parse_args()

    # Create dataset
    dataset = PolynomialDataset(args.polynomial_coeffs, noise_std=args.noise)
    # Format the polynomial as a string for display
    poly_terms = []
    for i, coeff in enumerate(args.polynomial_coeffs):
        power = len(args.polynomial_coeffs) - i - 1
        if coeff == 0:
            continue
        if power == 0:
            term = f"{coeff}"
        elif power == 1:
            term = f"{coeff}x"
        else:
            term = f"{coeff}x^{power}"
        poly_terms.append(term)
    poly_str = " + ".join(poly_terms).replace(" + -", " - ")

    print(f"Created polynomial dataset: {poly_str}")

    # Create model (1 input feature, specified hidden layers, 1 output)
    model = MLP(1, args.hidden_sizes, 1)
    print(
        f"Created MLP with architecture: 1 -> {' -> '.join(map(str, args.hidden_sizes))} -> 1")

    # Create trainer
    trainer = Trainer(
        model=model,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay
    )

    # Train model
    print(
        f"Training for {args.epochs} epochs with lr={args.lr}, batch_size={args.batch_size}")
    metrics = trainer.train(dataset, epochs=args.epochs)

    # Final evaluation
    final_loss = trainer.evaluate(dataset)
    print(f"Final evaluation loss: {final_loss:.6f}")


if __name__ == "__main__":
    main()

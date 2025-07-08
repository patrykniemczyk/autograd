# Neural Network Framework with Automatic Differentiation

A lightweight, educational neural network framework implemented from scratch with automatic differentiation. This framework has been comprehensively refactored to improve code quality, maintainability, and performance while preserving its educational value.

## Features

### Core Components
- **Variable.py**: Automatic differentiation engine with gradient computation
- **MLP.py**: Multi-layer perceptron with configurable architecture
- **AdamW.py**: AdamW optimizer with decoupled weight decay
- **Dataset.py**: Polynomial regression dataset with configurable generation
- **Trainer.py**: Training loop with advanced features

### Enhanced Features
- **Multiple activation functions**: ReLU, Tanh, Sigmoid
- **Multiple loss functions**: MSE, MAE
- **Multiple metrics**: MSE, MAE, R² score
- **Early stopping**: Prevent overfitting with patience-based stopping
- **Gradient clipping**: Numerical stability safeguards
- **Model persistence**: Save and load trained models
- **Configuration management**: JSON-based hyperparameter configuration
- **Professional logging**: Structured logging instead of print statements
- **Comprehensive error handling**: Meaningful error messages and validation

## Quick Start

### Basic Usage

```python
from Variable import Variable
from MLP import MLP
from Trainer import Trainer
from Dataset import PolynomialDataset

# Create dataset for y = x^2 - 1
dataset = PolynomialDataset([1, 0, -1], num_samples=1000)

# Create model
model = MLP(input_size=1, hidden_sizes=[16, 8], output_size=1, activation='tanh')

# Create trainer
trainer = Trainer(model, learning_rate=0.01, batch_size=32)

# Train model
metrics = trainer.train(dataset, epochs=100, early_stopping_patience=10)
```

### Command Line Interface

The framework includes a comprehensive CLI for training:

```bash
# Basic training
python Trainer.py --epochs 100 --hidden-sizes 16 16 --activation tanh

# Advanced training with early stopping
python Trainer.py \
    --epochs 200 \
    --hidden-sizes 32 16 8 \
    --activation sigmoid \
    --lr 0.001 \
    --early-stopping-patience 15 \
    --loss-fn mae \
    --save-model model.pkl \
    --save-config config.json

# Load and continue training
python Trainer.py --load-model model.pkl --epochs 50
```

### Configuration System

Use JSON configuration files for reproducible experiments:

```python
from Config import get_default_config

# Get default configuration
config = get_default_config()

# Modify as needed
config.model.hidden_sizes = [32, 16]
config.model.activation = 'tanh'
config.training.early_stopping_patience = 20

# Save configuration
config.save('my_experiment.json')

# Load configuration
config = Config.load('my_experiment.json')
```

### Model Persistence

Save and load trained models with full state:

```python
from ModelUtils import save_model, load_model

# Save model
save_model(model, 'trained_model.pkl', optimizer, config, metadata)

# Load model
model, optimizer_state, config, metadata = load_model('trained_model.pkl')
```

## Architecture

### Variable Class
The core of the automatic differentiation system:
- Tracks computational graphs
- Implements reverse-mode autodiff
- Supports basic operations (+, -, *, /, **)
- Includes activation functions (ReLU, Tanh, Sigmoid)
- Provides gradient clipping for numerical stability

### MLP Class
Multi-layer perceptron implementation:
- Configurable architecture
- Multiple activation functions
- Proper weight initialization (He initialization)
- Parameter counting and introspection

### Trainer Class
Advanced training functionality:
- Multiple loss functions and metrics
- Early stopping with configurable patience
- Comprehensive logging
- Batch processing with gradient clipping
- Evaluation on multiple metrics

## Code Quality Improvements

### Type Hints
All functions and classes include comprehensive type annotations:
```python
def train(
    self,
    dataset: Dataset,
    epochs: int = 100,
    early_stopping_patience: Optional[int] = None
) -> List[Dict[str, float]]:
```

### Error Handling
Meaningful error messages with proper validation:
```python
if input_size <= 0:
    raise ValueError(f"input_size must be positive, got {input_size}")
```

### Documentation
Comprehensive docstrings with examples:
```python
def relu(self) -> "Variable":
    """
    ReLU activation function with gradient tracking.
    
    Returns:
        New Variable with ReLU applied: max(0, self)
        
    Examples:
        >>> x = Variable(-2.0)
        >>> y = x.relu()  # y.data = 0.0
    """
```

### Numerical Stability
Safeguards against common numerical issues:
- Gradient clipping
- Overflow/underflow detection
- Numerically stable sigmoid implementation
- Division by zero protection

## Testing

Run the unit tests to verify functionality:

```bash
python test_framework.py
```

The test suite covers:
- Variable operations and gradient computation
- MLP architecture and forward pass
- Dataset generation and batching
- Loss functions and metrics
- Configuration validation
- Trainer functionality

## Examples

### Custom Polynomial
```python
# Create dataset for y = 2x³ - x² + 3x - 1
dataset = PolynomialDataset([2, -1, 3, -1], noise_std=0.05)
```

### Different Activations
```python
# ReLU network (default)
model_relu = MLP(1, [16], 1, activation='relu')

# Tanh network (better for regression)
model_tanh = MLP(1, [16], 1, activation='tanh')

# Sigmoid network
model_sigmoid = MLP(1, [16], 1, activation='sigmoid')
```

### Early Stopping
```python
trainer.train(
    dataset, 
    epochs=1000,
    early_stopping_patience=20,
    early_stopping_min_delta=1e-6
)
```

## Performance Optimizations

- **Gradient clipping**: Prevents exploding gradients
- **Numerically stable operations**: Avoids overflow/underflow
- **Efficient batch processing**: Optimized training loops
- **Memory management**: Proper gradient zeroing

## Educational Value

This framework is designed to be educational while being production-ready:
- Clear, readable code structure
- Comprehensive documentation with examples
- Step-by-step gradient computation
- Modular design for easy understanding
- No external dependencies beyond matplotlib

## License

This project is open source and available under the MIT License.
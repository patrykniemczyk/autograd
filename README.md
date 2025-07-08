# autograd

A lightweight neural network framework implemented from scratch with automatic differentiation.

## Quick Start

### Basic Usage

```python
from variable import Variable
from mlp import MLP
from trainer import Trainer
from dataset import PolynomialDataset

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
python trainer.py --epochs 100 --hidden-sizes 16 16 --activation tanh

# Advanced training with early stopping
python trainer.py \
    --epochs 200 \
    --hidden-sizes 32 16 8 \
    --activation sigmoid \
    --lr 0.001 \
    --early-stopping-patience 15 \
    --loss-fn mae \
    --save-model model.pkl \
    --save-config config.json

# Load and continue training
python trainer.py --load-model model.pkl --epochs 50
```

### Configuration System

Use JSON configuration files for reproducible experiments:

```python
from config import get_default_config

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
from model_utils import save_model, load_model

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
- Supports basic operations (+, -, \*, /, \*\*)
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

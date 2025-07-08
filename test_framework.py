"""
Unit tests for the neural network framework.

This module provides basic unit tests for core functionality to ensure
the refactored code works correctly.
"""

import math
import unittest
from Variable import Variable
from MLP import MLP, get_activation
from AdamW import AdamW
from Dataset import PolynomialDataset
from Trainer import Trainer, mean_squared_error, mean_absolute_error, r2_score
from Config import get_default_config, Config, ModelConfig


class TestVariable(unittest.TestCase):
    """Test cases for Variable class."""
    
    def test_basic_operations(self):
        """Test basic arithmetic operations."""
        x = Variable(2.0)
        y = Variable(3.0)
        
        # Test addition
        z = x + y
        self.assertEqual(z.data, 5.0)
        
        # Test multiplication
        z = x * y
        self.assertEqual(z.data, 6.0)
        
        # Test power
        z = x ** 2
        self.assertEqual(z.data, 4.0)
    
    def test_gradient_computation(self):
        """Test gradient computation."""
        x = Variable(2.0)
        y = Variable(3.0)
        z = x * y + x
        z.backward()
        
        self.assertEqual(x.grad, 4.0)  # dz/dx = y + 1 = 3 + 1 = 4
        self.assertEqual(y.grad, 2.0)  # dz/dy = x = 2
    
    def test_activation_functions(self):
        """Test activation functions."""
        x = Variable(0.0)
        
        # Test ReLU
        relu_out = x.relu()
        self.assertEqual(relu_out.data, 0.0)
        
        # Test Tanh
        tanh_out = x.tanh()
        self.assertEqual(tanh_out.data, 0.0)
        
        # Test Sigmoid
        sigmoid_out = x.sigmoid()
        self.assertEqual(sigmoid_out.data, 0.5)
    
    def test_error_handling(self):
        """Test error handling."""
        with self.assertRaises(ValueError):
            Variable(float('nan'))
        
        with self.assertRaises(ValueError):
            Variable(float('inf'))
        
        # Test division by zero
        x = Variable(1.0)
        y = Variable(0.0)
        with self.assertRaises(ValueError):
            z = x / y
    
    def test_gradient_clipping(self):
        """Test gradient clipping."""
        x = Variable(1.0)
        x.grad = 10.0
        x.clip_grad(5.0)
        self.assertEqual(x.grad, 5.0)


class TestMLP(unittest.TestCase):
    """Test cases for MLP class."""
    
    def test_model_creation(self):
        """Test model creation with different configurations."""
        model = MLP(input_size=2, hidden_sizes=[4], output_size=1)
        self.assertEqual(model.input_size, 2)
        self.assertEqual(model.output_size, 1)
        self.assertEqual(model.hidden_sizes, [4])
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = MLP(input_size=2, hidden_sizes=[4], output_size=1)
        inputs = [Variable(1.0), Variable(2.0)]
        outputs = model(inputs)
        
        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(outputs[0], Variable)
    
    def test_parameter_count(self):
        """Test parameter counting."""
        model = MLP(input_size=2, hidden_sizes=[4], output_size=1)
        # Input to hidden: 2*4 + 4 = 12
        # Hidden to output: 4*1 + 1 = 5
        # Total: 17
        self.assertEqual(model.num_parameters(), 17)
    
    def test_activation_factory(self):
        """Test activation function factory."""
        relu = get_activation('relu')
        tanh = get_activation('tanh')
        sigmoid = get_activation('sigmoid')
        
        self.assertIsNotNone(relu)
        self.assertIsNotNone(tanh)
        self.assertIsNotNone(sigmoid)
        
        with self.assertRaises(ValueError):
            get_activation('invalid')


class TestDataset(unittest.TestCase):
    """Test cases for Dataset class."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = PolynomialDataset([1, 0, -1], num_samples=100, seed=42)
        self.assertEqual(len(dataset), 80)  # 100 * (1 - 0.2) = 80
    
    def test_batch_generation(self):
        """Test batch generation."""
        dataset = PolynomialDataset([1, 0, -1], num_samples=100, seed=42)
        batch_x, batch_y = dataset.get_batch(10)
        
        self.assertEqual(len(batch_x), 10)
        self.assertEqual(len(batch_y), 10)
    
    def test_polynomial_evaluation(self):
        """Test polynomial evaluation."""
        dataset = PolynomialDataset([1, 0, -1], num_samples=100, seed=42)
        # For polynomial x^2 - 1 at x=2: 2^2 - 1 = 3
        result = dataset._evaluate_polynomial(2.0)
        self.assertEqual(result, 3.0)


class TestLossFunctions(unittest.TestCase):
    """Test cases for loss functions."""
    
    def test_mse(self):
        """Test mean squared error."""
        pred = [Variable(1.0), Variable(2.0)]
        target = [Variable(1.0), Variable(2.0)]
        
        mse = mean_squared_error(pred, target)
        self.assertEqual(mse.data, 0.0)
    
    def test_mae(self):
        """Test mean absolute error."""
        pred = [Variable(1.0), Variable(2.0)]
        target = [Variable(1.0), Variable(2.0)]
        
        mae = mean_absolute_error(pred, target)
        self.assertEqual(mae.data, 0.0)
    
    def test_r2_score(self):
        """Test R² score."""
        pred = [Variable(1.0), Variable(2.0), Variable(3.0)]
        target = [Variable(1.0), Variable(2.0), Variable(3.0)]
        
        r2 = r2_score(pred, target)
        self.assertEqual(r2, 1.0)  # Perfect prediction


class TestConfig(unittest.TestCase):
    """Test cases for configuration system."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = get_default_config()
        self.assertEqual(config.model.input_size, 1)
        self.assertEqual(config.model.activation, 'relu')
    
    def test_config_validation(self):
        """Test configuration validation."""
        with self.assertRaises(ValueError):
            ModelConfig(input_size=0, hidden_sizes=[4], output_size=1)
        
        with self.assertRaises(ValueError):
            ModelConfig(input_size=1, hidden_sizes=[-1], output_size=1)


class TestTrainer(unittest.TestCase):
    """Test cases for Trainer class."""
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        model = MLP(input_size=1, hidden_sizes=[4], output_size=1)
        trainer = Trainer(model, learning_rate=0.01)
        
        self.assertEqual(trainer.learning_rate, 0.01)
        self.assertIsInstance(trainer.optimizer, AdamW)
    
    def test_evaluation(self):
        """Test model evaluation."""
        model = MLP(input_size=1, hidden_sizes=[4], output_size=1)
        trainer = Trainer(model, learning_rate=0.01)
        dataset = PolynomialDataset([1, 0, -1], num_samples=100, seed=42)
        
        metrics = trainer.evaluate(dataset, metrics=['mse', 'mae', 'r2'])
        
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)


def run_tests():
    """Run all unit tests."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()
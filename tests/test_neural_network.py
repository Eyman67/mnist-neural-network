import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_network import NeuralNetwork
from utils import vectorized_result

class TestNeuralNetwork(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.network = NeuralNetwork([2, 3, 1])
    
    def test_initialization(self):
        """Test that network initializes correctly"""
        self.assertEqual(self.network.num_layers, 3)
        self.assertEqual(self.network.sizes, [2, 3, 1])
        self.assertEqual(len(self.network.weights), 2)
        self.assertEqual(len(self.network.biases), 2)
    
    def test_sigmoid_function(self):
        """Test sigmoid activation function"""
        # Test known values
        self.assertAlmostEqual(self.network.sigmoid(0), 0.5, places=5)
        self.assertAlmostEqual(self.network.sigmoid(np.inf), 1.0, places=5)
        self.assertAlmostEqual(self.network.sigmoid(-np.inf), 0.0, places=5)
    
    def test_feedforward(self):
        """Test feedforward propagation"""
        input_vector = np.array([[1.0], [0.5]])
        output = self.network.feedforward(input_vector)
        
        # Output should be a column vector with shape (1, 1)
        self.assertEqual(output.shape, (1, 1))
        # Output should be between 0 and 1 (sigmoid output)
        self.assertTrue(0 <= output[0, 0] <= 1)
    
    def test_vectorized_result(self):
        """Test vectorized result function"""
        result = vectorized_result(5)
        expected = np.zeros((10, 1))
        expected[5] = 1.0
        
        np.testing.assert_array_equal(result, expected)
    
    def test_cost_derivative(self):
        """Test cost function derivative"""
        output = np.array([[0.8], [0.2]])
        target = np.array([[1.0], [0.0]])
        
        derivative = self.network.cost_derivative(output, target)
        expected = np.array([[-0.2], [0.2]])
        
        np.testing.assert_array_almost_equal(derivative, expected)

class TestMNISTDimensions(unittest.TestCase):
    
    def test_mnist_network_dimensions(self):
        """Test that MNIST network has correct dimensions"""
        mnist_network = NeuralNetwork([784, 30, 10])
        
        self.assertEqual(mnist_network.sizes, [784, 30, 10])
        self.assertEqual(mnist_network.weights[0].shape, (30, 784))
        self.assertEqual(mnist_network.weights[1].shape, (10, 30))
        self.assertEqual(mnist_network.biases[0].shape, (30, 1))
        self.assertEqual(mnist_network.biases[1].shape, (10, 1))

if __name__ == '__main__':
    unittest.main()
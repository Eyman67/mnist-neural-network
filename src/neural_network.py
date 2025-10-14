import numpy as np
from typing import List, Tuple
import pickle

class NeuralNetwork:
    """
    A feedforward neural network implementation with backpropagation
    and stochastic gradient descent
    """
    
    def __init__(self, sizes: List[int]):
        """
        Initialize the neural network
        
        Args:
            sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        # Initialize weights and biases using Gaussian distribution
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_prime(self, z: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function"""
        sig = self.sigmoid(z)
        return sig * (1 - sig)
    
    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the network
        
        Args:
            a: Input vector (column vector)
            
        Returns:
            Output of the network
        """
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data: List[Tuple], epochs: int, mini_batch_size: int, 
            eta: float, test_data: List[Tuple] = None) -> None:
        """
        Train the network using stochastic gradient descent
        
        Args:
            training_data: List of (x, y) tuples representing training inputs and outputs
            epochs: Number of epochs to train for
            mini_batch_size: Size of mini-batches for SGD
            eta: Learning rate
            test_data: Optional test data for evaluation
        """
        training_data = list(training_data)
        n = len(training_data)
        
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        
        for j in range(epochs):
            # Shuffle training data
            np.random.shuffle(training_data)
            
            # Create mini-batches
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            
            # Update network for each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            # Evaluate on test data if provided
            if test_data:
                accuracy = self.evaluate(test_data)
                print(f"Epoch {j+1}: {accuracy}/{n_test} ({100*accuracy/n_test:.2f}%)")
            else:
                print(f"Epoch {j+1} complete")
    
    def update_mini_batch(self, mini_batch: List[Tuple], eta: float) -> None:
        """
        Update network weights and biases using backpropagation
        for a single mini-batch
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Update weights and biases
        self.weights = [w - (eta/len(mini_batch)) * nw 
                       for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb 
                      for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backpropagation algorithm to compute gradients
        
        Returns:
            Tuple of (nabla_b, nabla_w) representing gradients for biases and weights
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Feedforward
        activation = x
        activations = [x]  # Store all activations
        zs = []  # Store all z vectors
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)
    
    def cost_derivative(self, output_activations: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the vector of partial derivatives for the output activations"""
        return (output_activations - y)
    
    def evaluate(self, test_data: List[Tuple]) -> int:
        """
        Return the number of test inputs for which the neural network
        outputs the correct result
        """
        test_results = [(np.argmax(self.feedforward(x)), y) 
                       for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def save_model(self, filename: str) -> None:
        """Save the trained model to a file"""
        model_data = {
            'sizes': self.sizes,
            'weights': self.weights,
            'biases': self.biases
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename: str) -> None:
        """Load a trained model from a file"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.sizes = model_data['sizes']
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.num_layers = len(self.sizes)
        print(f"Model loaded from {filename}")
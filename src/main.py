"""
MNIST Handwritten Digit Recognition using Neural Networks
A complete implementation of a feedforward neural network trained with
stochastic gradient descent to classify handwritten digits.
"""

import numpy as np
import os
import sys
from typing import Optional

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import MNISTLoader
from neural_network import NeuralNetwork
from utils import prepare_training_data, prepare_test_data, plot_sample_predictions

def main():
    """Main function to train and evaluate the neural network"""
    
    print("MNIST Handwritten Digit Recognition")
    print("===================================")
    
    # Initialize data loader
    print("\n1. Loading MNIST dataset...")
    loader = MNISTLoader()
    (train_images, train_labels), (test_images, test_labels) = loader.load_data()
    
    print(f"Training set: {len(train_images)} images")
    print(f"Test set: {len(test_images)} images")
    
    # Prepare data for neural network
    print("\n2. Preparing data for neural network...")
    training_data = prepare_training_data(train_images, train_labels)
    test_data = prepare_test_data(test_images, test_labels)
    
    # Initialize neural network
    print("\n3. Initializing neural network...")
    # Network architecture: 784 input neurons, 30 hidden neurons, 10 output neurons
    network = NeuralNetwork([784, 30, 10])
    
    # Training parameters
    epochs = 30
    mini_batch_size = 10
    learning_rate = 3.0
    
    print(f"Network architecture: {network.sizes}")
    print(f"Training parameters:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Mini-batch size: {mini_batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    
    # Train the network
    print("\n4. Training neural network...")
    print("-" * 50)
    
    network.SGD(
        training_data=training_data,
        epochs=epochs,
        mini_batch_size=mini_batch_size,
        eta=learning_rate,
        test_data=test_data
    )
    
    # Final evaluation
    print("\n5. Final evaluation...")
    final_accuracy = network.evaluate(test_data)
    print(f"Final test accuracy: {final_accuracy}/{len(test_data)} "
          f"({100*final_accuracy/len(test_data):.2f}%)")
    
    # Save the trained model
    print("\n6. Saving trained model...")
    os.makedirs("models", exist_ok=True)
    model_path = "models/mnist_nn_model.pkl"
    network.save_model(model_path)
    
    # Display sample predictions
    print("\n7. Displaying sample predictions...")
    try:
        import matplotlib.pyplot as plt
        plot_sample_predictions(network, test_images, test_labels)
    except ImportError:
        print("Matplotlib not available for plotting. Install with: pip install matplotlib")
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {model_path}")

def predict_single_digit(model_path: str, image_path: Optional[str] = None):
    """
    Load trained model and predict a single digit
    
    Args:
        model_path: Path to the saved model
        image_path: Optional path to image file
    """
    # Load the trained model
    network = NeuralNetwork([784, 30, 10])
    network.load_model(model_path)
    
    if image_path:
        # Load and preprocess custom image
        try:
            from PIL import Image
            img = Image.open(image_path).convert('L')
            img = img.resize((28, 28))
            image_array = np.array(img) / 255.0
            image_vector = image_array.reshape(784, 1)
            
            # Make prediction
            prediction = network.feedforward(image_vector)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            print(f"Predicted digit: {predicted_digit}")
            print(f"Confidence: {confidence:.4f}")
            
        except ImportError:
            print("PIL not available. Install with: pip install pillow")
        except Exception as e:
            print(f"Error processing image: {e}")
    else:
        # Use a random test image
        loader = MNISTLoader()
        (_, _), (test_images, test_labels) = loader.load_data()
        
        # Pick a random test image
        idx = np.random.randint(0, len(test_images))
        image = test_images[idx]
        true_label = test_labels[idx]
        
        # Make prediction
        prediction = network.feedforward(image.reshape(784, 1))
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        print(f"True digit: {true_label}")
        print(f"Predicted digit: {predicted_digit}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Correct: {'Yes' if predicted_digit == true_label else 'No'}")

if __name__ == "__main__":
    # Check if we want to train or predict
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        model_path = "models/mnist_nn_model.pkl"
        if os.path.exists(model_path):
            image_path = sys.argv[2] if len(sys.argv) > 2 else None
            predict_single_digit(model_path, image_path)
        else:
            print("No trained model found. Please train the model first by running: python main.py")
    else:
        # Train the model
        main()
#morchidy
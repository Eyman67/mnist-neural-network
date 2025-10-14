import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def vectorized_result(j: int) -> np.ndarray:
    """
    Return a 10-dimensional unit vector with a 1.0 in the j-th position
    and zeroes elsewhere. Used to convert digit labels to network output format.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def prepare_training_data(images: np.ndarray, labels: np.ndarray) -> List[Tuple]:
    """
    Prepare training data in the format expected by the neural network
    
    Args:
        images: Array of flattened images
        labels: Array of digit labels (0-9)
    
    Returns:
        List of (image, vectorized_label) tuples
    """
    training_inputs = [np.reshape(x, (784, 1)) for x in images]
    training_results = [vectorized_result(y) for y in labels]
    return list(zip(training_inputs, training_results))

def prepare_test_data(images: np.ndarray, labels: np.ndarray) -> List[Tuple]:
    """
    Prepare test data in the format expected by the neural network
    
    Args:
        images: Array of flattened images
        labels: Array of digit labels (0-9)
    
    Returns:
        List of (image, label) tuples
    """
    test_inputs = [np.reshape(x, (784, 1)) for x in images]
    return list(zip(test_inputs, labels))

def plot_digit(image: np.ndarray, label: int = None) -> None:
    """
    Plot a single MNIST digit
    
    Args:
        image: Flattened 784-dimensional image array
        label: Optional label for the image
    """
    plt.figure(figsize=(3, 3))
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.axis('off')
    if label is not None:
        plt.title(f'Label: {label}')
    plt.show()

def plot_training_progress(accuracies: List[float]) -> None:
    """
    Plot training accuracy over epochs
    
    Args:
        accuracies: List of accuracy values for each epoch
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies) + 1), accuracies, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Progress')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_sample_predictions(network, test_images: np.ndarray, 
                          test_labels: np.ndarray, num_samples: int = 10) -> None:
    """
    Plot sample predictions from the trained network
    
    Args:
        network: Trained neural network
        test_images: Test images array
        test_labels: Test labels array
        num_samples: Number of samples to display
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    
    indices = np.random.choice(len(test_images), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        row = i // 5
        col = i % 5
        
        image = test_images[idx]
        true_label = test_labels[idx]
        
        # Get prediction
        prediction = network.feedforward(image.reshape(784, 1))
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Plot
        axes[row, col].imshow(image.reshape(28, 28), cmap='gray')
        axes[row, col].axis('off')
        color = 'green' if predicted_label == true_label else 'red'
        axes[row, col].set_title(f'True: {true_label}, Pred: {predicted_label}\n'
                               f'Confidence: {confidence:.2f}', 
                               color=color, fontsize=10)
    
    plt.tight_layout()
    plt.show()
#morchidy
import numpy as np
import gzip
import urllib.request
import os
import ssl
from typing import Tuple

class MNISTLoader:
    """
    MNIST dataset loader with automatic download capability
    """
    
    def __init__(self, data_dir: str = "data/mnist"):
        self.data_dir = data_dir
        # Try multiple mirror sources
        self.base_urls = [
            "https://storage.googleapis.com/cvdf-datasets/mnist/",
            "http://yann.lecun.com/exdb/mnist/",
            "https://ossci-datasets.s3.amazonaws.com/mnist/"
        ]
        self.files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz'
        }
        
    def download_data(self) -> None:
        """Download MNIST data if not already present"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create an SSL context that doesn't verify certificates (for some mirrors)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        for file_key, filename in self.files.items():
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                
                downloaded = False
                for base_url in self.base_urls:
                    try:
                        url = base_url + filename
                        print(f"  Trying: {url}")
                        
                        # Create a request with headers
                        req = urllib.request.Request(
                            url, 
                            headers={
                                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
                            }
                        )
                        
                        # Try with SSL context for HTTPS URLs
                        if url.startswith('https'):
                            opener = urllib.request.build_opener(
                                urllib.request.HTTPSHandler(context=ssl_context)
                            )
                            urllib.request.install_opener(opener)
                        
                        urllib.request.urlretrieve(url, filepath)
                        print(f"  ✓ Downloaded {filename} from {base_url}")
                        downloaded = True
                        break
                        
                    except Exception as e:
                        print(f"  ✗ Failed to download from {base_url}: {e}")
                        continue
                
                if not downloaded:
                    print(f"\n❌ Could not download {filename} from any source.")
                    print("Please download MNIST data manually:")
                    print("1. Go to http://yann.lecun.com/exdb/mnist/")
                    print(f"2. Download {filename}")
                    print(f"3. Place it in {self.data_dir}/")
                    raise FileNotFoundError(f"Could not download {filename}")
    
    def load_images(self, filename: str) -> np.ndarray:
        """Load images from MNIST file"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        with gzip.open(filepath, 'rb') as f:
            # Read magic number and dimensions
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            
            # Verify magic number
            expected_magic = 2051
            if magic != expected_magic:
                raise ValueError(f"Invalid magic number: {magic}, expected {expected_magic}")
            
            # Read image data
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows * cols)
            
        return images.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    def load_labels(self, filename: str) -> np.ndarray:
        """Load labels from MNIST file"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        with gzip.open(filepath, 'rb') as f:
            # Read magic number and count
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')
            
            # Verify magic number
            expected_magic = 2049
            if magic != expected_magic:
                raise ValueError(f"Invalid magic number: {magic}, expected {expected_magic}")
            
            # Read label data
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            
        return labels
    
    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load complete MNIST dataset
        
        Returns:
            training_data: (train_images, train_labels)
            test_data: (test_images, test_labels)
        """
        try:
            self.download_data()
        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            print("\nAlternative: You can use a different approach to get MNIST data.")
            print("Would you like to use a simpler method? (y/n)")
            response = input().lower().strip()
            if response == 'y':
                return self._load_with_fallback()
            else:
                raise
        
        # Load training data
        train_images = self.load_images(self.files['train_images'])
        train_labels = self.load_labels(self.files['train_labels'])
        
        # Load test data
        test_images = self.load_images(self.files['test_images'])
        test_labels = self.load_labels(self.files['test_labels'])
        
        print(f"✓ Successfully loaded MNIST dataset")
        print(f"  Training images: {train_images.shape}")
        print(f"  Training labels: {train_labels.shape}")
        print(f"  Test images: {test_images.shape}")
        print(f"  Test labels: {test_labels.shape}")
        
        return (train_images, train_labels), (test_images, test_labels)
    
    def _load_with_fallback(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Fallback method using synthetic data for demonstration
        """
        print("Creating synthetic MNIST-like data for demonstration...")
        
        # Create synthetic data that mimics MNIST structure
        np.random.seed(42)  # For reproducibility
        
        # Training data: 1000 samples
        train_images = np.random.rand(1000, 784).astype(np.float32)
        train_labels = np.random.randint(0, 10, 1000).astype(np.uint8)
        
        # Test data: 200 samples
        test_images = np.random.rand(200, 784).astype(np.float32)
        test_labels = np.random.randint(0, 10, 200).astype(np.uint8)
        
        print("✓ Created synthetic dataset")
        print("  (Note: This is random data for testing purposes)")
        print(f"  Training images: {train_images.shape}")
        print(f"  Training labels: {train_labels.shape}")
        print(f"  Test images: {test_images.shape}")
        print(f"  Test labels: {test_labels.shape}")
        
        return (train_images, train_labels), (test_images, test_labels)
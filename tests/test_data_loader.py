import unittest
import numpy as np
import sys
import os
import tempfile
import shutil

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import MNISTLoader

class TestMNISTLoader(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.loader = MNISTLoader(data_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test that loader initializes correctly"""
        self.assertEqual(self.loader.data_dir, self.temp_dir)
        self.assertIn('train_images', self.loader.files)
        self.assertIn('train_labels', self.loader.files)
        self.assertIn('test_images', self.loader.files)
        self.assertIn('test_labels', self.loader.files)
    
    def test_data_dir_creation(self):
        """Test that data directory is created"""
        new_temp_dir = os.path.join(self.temp_dir, 'new_mnist_dir')
        new_loader = MNISTLoader(data_dir=new_temp_dir)
        new_loader.download_data()
        
        self.assertTrue(os.path.exists(new_temp_dir))

class TestDataIntegrity(unittest.TestCase):
    """Integration tests that require actual MNIST data"""
    
    def test_data_loading_integration(self):
        """Test actual data loading (requires internet connection)"""
        try:
            loader = MNISTLoader()
            (train_images, train_labels), (test_images, test_labels) = loader.load_data()
            
            # Test data shapes
            self.assertEqual(train_images.shape, (60000, 784))
            self.assertEqual(train_labels.shape, (60000,))
            self.assertEqual(test_images.shape, (10000, 784))
            self.assertEqual(test_labels.shape, (10000,))
            
            # Test data ranges
            self.assertTrue(np.all(train_images >= 0.0))
            self.assertTrue(np.all(train_images <= 1.0))
            self.assertTrue(np.all(train_labels >= 0))
            self.assertTrue(np.all(train_labels <= 9))
            
        except Exception as e:
            self.skipTest(f"Could not download MNIST data: {e}")

if __name__ == '__main__':
    unittest.main()
#morchidy
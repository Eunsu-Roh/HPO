"""MNIST data loader with support for both CNN and traditional ML models."""

import numpy as np
from torchvision import datasets
from sklearn.model_selection import train_test_split


class MNISTDataLoader:
    """Load and preprocess MNIST dataset for various model types."""
    
    def __init__(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                 normalize=True, random_state=42):
        """
        Initialize MNIST data loader.
        
        Args:
            train_ratio: Proportion of data for training (default: 0.7)
            val_ratio: Proportion of data for validation (default: 0.15)
            test_ratio: Proportion of data for testing (default: 0.15)
            normalize: Whether to normalize pixel values to [0, 1] (default: True)
            random_state: Random seed for reproducibility (default: 42)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.normalize = normalize
        self.random_state = random_state
        
        # Load data
        self._load_and_split_data()
    
    def _load_and_split_data(self):
        """Load MNIST and split into train/val/test sets."""
        # Load MNIST dataset using torchvision
        train_dataset = datasets.MNIST(root='./data', train=True, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True)
        
        # Convert to numpy arrays
        X_train_full = train_dataset.data.numpy()
        y_train_full = train_dataset.targets.numpy()
        X_test_orig = test_dataset.data.numpy()
        y_test_orig = test_dataset.targets.numpy()
        
        # Combine training and test sets for custom split
        X_all = np.concatenate([X_train_full, X_test_orig], axis=0)
        y_all = np.concatenate([y_train_full, y_test_orig], axis=0)
        
        # First split: separate test set
        test_size = self.test_ratio
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_all, y_all, test_size=test_size, 
            random_state=self.random_state, stratify=y_all
        )
        
        # Second split: separate train and validation
        val_size_adjusted = self.val_ratio / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=self.random_state, stratify=y_temp
        )
        
        # Store raw data
        self.X_train_raw = X_train
        self.X_val_raw = X_val
        self.X_test_raw = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        print(f"Data split complete:")
        print(f"  Train: {len(X_train)} samples ({self.train_ratio*100:.1f}%)")
        print(f"  Val:   {len(X_val)} samples ({self.val_ratio*100:.1f}%)")
        print(f"  Test:  {len(X_test)} samples ({self.test_ratio*100:.1f}%)")
    
    def get_data_for_cnn(self):
        """
        Get data in format suitable for CNN models (TensorFlow/Keras).
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
            where X has shape (n_samples, 28, 28, 1)
        """
        # Reshape to add channel dimension
        X_train = self.X_train_raw.reshape(-1, 28, 28, 1)
        X_val = self.X_val_raw.reshape(-1, 28, 28, 1)
        X_test = self.X_test_raw.reshape(-1, 28, 28, 1)
        
        # Normalize to [0, 1] if requested
        if self.normalize:
            X_train = X_train.astype('float32') / 255.0
            X_val = X_val.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
        
        return X_train, self.y_train, X_val, self.y_val, X_test, self.y_test
    
    def get_data_for_ml(self):
        """
        Get data in format suitable for traditional ML models (RandomForest, XGBoost).
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
            where X has shape (n_samples, 784) - flattened images
        """
        # Flatten images
        X_train = self.X_train_raw.reshape(-1, 28 * 28)
        X_val = self.X_val_raw.reshape(-1, 28 * 28)
        X_test = self.X_test_raw.reshape(-1, 28 * 28)
        
        # Normalize to [0, 1] if requested
        if self.normalize:
            X_train = X_train.astype('float32') / 255.0
            X_val = X_val.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
        
        return X_train, self.y_train, X_val, self.y_val, X_test, self.y_test
    
    def get_combined_train_val_for_cv(self, for_cnn=False):
        """
        Get combined train+val data for cross-validation.
        
        Args:
            for_cnn: If True, return CNN format; otherwise ML format
            
        Returns:
            Tuple of (X, y) combining train and validation sets
        """
        if for_cnn:
            X_train = self.X_train_raw.reshape(-1, 28, 28, 1)
            X_val = self.X_val_raw.reshape(-1, 28, 28, 1)
        else:
            X_train = self.X_train_raw.reshape(-1, 28 * 28)
            X_val = self.X_val_raw.reshape(-1, 28 * 28)
        
        X = np.concatenate([X_train, X_val], axis=0)
        y = np.concatenate([self.y_train, self.y_val], axis=0)
        
        # Normalize if requested
        if self.normalize:
            X = X.astype('float32') / 255.0
        
        return X, y
    
    def get_info(self):
        """Get information about the dataset."""
        return {
            'total_samples': len(self.X_train_raw) + len(self.X_val_raw) + len(self.X_test_raw),
            'train_samples': len(self.X_train_raw),
            'val_samples': len(self.X_val_raw),
            'test_samples': len(self.X_test_raw),
            'image_shape': (28, 28),
            'num_classes': 10,
            'class_names': list(range(10)),
            'normalized': self.normalize
        }

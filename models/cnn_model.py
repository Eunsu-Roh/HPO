"""CNN model builder for MNIST classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """PyTorch CNN model with configurable architecture."""
    
    def __init__(self, num_conv_layers=2, conv_filters=None, dense_units=128,
                 dropout_rate=0.3, num_classes=10):
        """
        Initialize CNN model.
        
        Args:
            num_conv_layers: Number of convolutional layers (2-4)
            conv_filters: List of filter sizes for each conv layer
            dense_units: Number of units in dense layer
            dropout_rate: Dropout rate (0.0-0.5)
            num_classes: Number of output classes (default: 10)
        """
        super(CNNModel, self).__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Default filter configuration
        if conv_filters is None:
            if num_conv_layers == 2:
                conv_filters = [32, 64]
            elif num_conv_layers == 3:
                conv_filters = [32, 64, 128]
            else:
                conv_filters = [32, 64, 128, 256]
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
        in_channels = 1
        for filters in conv_filters:
            self.conv_layers.append(
                nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
            )
            in_channels = filters
        
        # Calculate flattened size (28x28 -> after pooling layers)
        # Each pool layer reduces by 2, so 28 / (2^num_layers)
        feature_size = 28 // (2 ** num_conv_layers)
        flattened_size = conv_filters[-1] * feature_size * feature_size
        
        # Dense layers
        self.fc1 = nn.Linear(flattened_size, dense_units)
        self.dropout_fc = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.fc2 = nn.Linear(dense_units, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 1, 28, 28)
            
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Convolutional layers
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            x = self.pool(x)
            if self.dropout_conv is not None:
                x = self.dropout_conv(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        if self.dropout_fc is not None:
            x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x


class CNNModelBuilder:
    """Builder class for CNN models."""
    
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build(self, num_conv_layers=2, conv_filters=None, dense_units=128,
              dropout_rate=0.3, learning_rate=0.001, activation='relu'):
        """Build and return a CNN model."""
        model = CNNModel(
            num_conv_layers=num_conv_layers,
            conv_filters=conv_filters,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            num_classes=self.num_classes
        )
        return model, learning_rate


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10, **params):
    """
    Convenience function to build a CNN model.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
        **params: Model parameters (num_conv_layers, conv_filters, dense_units,
                 dropout_rate, learning_rate, activation)
                 
    Returns:
        Tuple of (PyTorch model, learning_rate)
    """
    builder = CNNModelBuilder(input_shape=input_shape, num_classes=num_classes)
    
    # Extract parameters with defaults
    num_conv_layers = params.get('num_conv_layers', 2)
    conv_filters = params.get('conv_filters', None)
    dense_units = params.get('dense_units', 128)
    dropout_rate = params.get('dropout_rate', 0.3)
    learning_rate = params.get('learning_rate', 0.001)
    
    return builder.build(
        num_conv_layers=num_conv_layers,
        conv_filters=conv_filters,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

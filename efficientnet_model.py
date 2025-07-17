"""
EfficientNet-B0 model for Alzheimer's disease classification.

This module implements the EfficientNet-B0 architecture for classifying
MRI brain scans into four categories of Alzheimer's disease severity.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, Any
import logging


class EfficientNetAlzheimer(nn.Module):
    """
    EfficientNet-B0 model for Alzheimer's disease classification.
    
    This model uses transfer learning with a pre-trained EfficientNet-B0
    backbone and a custom classification head for 4-class classification.
    
    Args:
        num_classes (int): Number of output classes (default: 4)
        pretrained (bool): Whether to use pre-trained weights (default: True)
        dropout_rate (float): Dropout rate for regularization (default: 0.2)
    """
    
    def __init__(
        self, 
        num_classes: int = 4, 
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        super(EfficientNetAlzheimer, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load pre-trained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Get the number of input features for the classifier
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier with our custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logging.info(f"EfficientNet-B0 model initialized with {num_classes} classes")
    
    def _initialize_weights(self):
        """Initialize the weights of the classification head."""
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone without classification.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Feature tensor of shape (batch_size, feature_dim)
        """
        # Extract features from all layers except the classifier
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        return features
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities using softmax.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Probability tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class indices.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Predicted class indices of shape (batch_size,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def freeze_backbone(self):
        """Freeze the backbone parameters for fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        logging.info("Backbone parameters frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone parameters."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True
        logging.info("Backbone parameters unfrozen")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information including parameter count and size.
        
        Returns:
            Dict[str, Any]: Model information dictionary
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'EfficientNet-B0',
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'dropout_rate': self.dropout_rate,
            'input_size': (224, 224),
            'model_size_mb': (total_params * 4) / (1024 * 1024)  # Approximate size in MB
        }


def create_model(
    num_classes: int = 4,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    device: Optional[str] = None
) -> EfficientNetAlzheimer:
    """
    Factory function to create and initialize the model.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pre-trained weights
        dropout_rate (float): Dropout rate for regularization
        device (str, optional): Device to move the model to
        
    Returns:
        EfficientNetAlzheimer: Initialized model
    """
    model = EfficientNetAlzheimer(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    
    if device is not None:
        model = model.to(device)
        logging.info(f"Model moved to device: {device}")
    
    return model


def load_model(
    checkpoint_path: str,
    num_classes: int = 4,
    device: Optional[str] = None
) -> EfficientNetAlzheimer:
    """
    Load a model from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        num_classes (int): Number of output classes
        device (str, optional): Device to load the model to
        
    Returns:
        EfficientNetAlzheimer: Loaded model
    """
    model = create_model(num_classes=num_classes, pretrained=False)
    
    # Load the state dict
    if device is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    if device is not None:
        model = model.to(device)
    
    model.eval()
    logging.info(f"Model loaded from {checkpoint_path}")
    
    return model


if __name__ == "__main__":
    # Example usage
    import torch
    
    # Create model
    model = create_model(num_classes=4, pretrained=True)
    
    # Print model info
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output (logits): {output}")
    
    # Test prediction
    probs = model.predict_proba(dummy_input)
    pred = model.predict(dummy_input)
    print(f"Probabilities: {probs}")
    print(f"Predicted class: {pred}")
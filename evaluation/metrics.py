"""Comprehensive evaluation metrics for MNIST classification."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score, 
    confusion_matrix,
    classification_report
)


def predict_labels(model, X, model_type='sklearn'):
    """
    Get predictions from different model types.
    
    Args:
        model: Trained model (sklearn, xgboost, or pytorch)
        X: Input data
        model_type: Type of model ('sklearn', 'xgboost', 'pytorch')
        
    Returns:
        Predicted labels as 1D array
    """
    if model_type in ['sklearn', 'xgboost']:
        # For sklearn and xgboost models
        y_pred = model.predict(X)
    else:  # pytorch
        # For PyTorch models
        model.eval()
        device = next(model.parameters()).device  # 모델이 있는 디바이스 가져오기
        
        with torch.no_grad():
            # Convert to tensor if needed
            if not isinstance(X, torch.Tensor):
                X_tensor = torch.FloatTensor(X)
            else:
                X_tensor = X.clone()
            
            # Ensure correct shape (batch, 1, 28, 28)
            if len(X_tensor.shape) == 2:  # (batch, 784)
                X_tensor = X_tensor.view(-1, 1, 28, 28)
            elif len(X_tensor.shape) == 3:  # (batch, 28, 28)
                X_tensor = X_tensor.unsqueeze(1)
            elif len(X_tensor.shape) == 4 and X_tensor.shape[-1] == 1:  # (batch, 28, 28, 1) - HWC
                X_tensor = X_tensor.permute(0, 3, 1, 2)  # -> (batch, 1, 28, 28) CHW
            
            # Batch 처리 (메모리 오버플로우 방지)
            batch_size = 256
            y_pred_list = []
            
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size].to(device)
                outputs = model(batch)
                batch_preds = torch.argmax(outputs, dim=1).cpu()
                y_pred_list.append(batch_preds)
                
                # 배치 처리 후 메모리 정리
                del batch, outputs
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            y_pred = torch.cat(y_pred_list).numpy()
    
    return y_pred


def calculate_all_metrics(y_true, y_pred):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(per_class_acc):
        metrics[f'class_{i}_accuracy'] = float(acc)
    
    return metrics, cm


def evaluate_model(model, X_test, y_test, model_type='sklearn'):
    """
    Evaluate a model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_type: Type of model ('sklearn', 'xgboost', 'pytorch')
        
    Returns:
        Tuple of (metrics_dict, confusion_matrix, classification_report)
    """
    # Get predictions
    y_pred = predict_labels(model, X_test, model_type)
    
    # Calculate metrics
    metrics, cm = calculate_all_metrics(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return metrics, cm, report


def print_metrics(metrics, model_name="Model", hpo_method="HPO"):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
        hpo_method: Name of the HPO method
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {model_name} - {hpo_method}")
    print(f"{'='*60}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"Precision (weight): {metrics['precision_weighted']:.4f}")
    print(f"Recall (macro):     {metrics['recall_macro']:.4f}")
    print(f"Recall (weighted):  {metrics['recall_weighted']:.4f}")
    print(f"F1 Score (macro):   {metrics['f1_macro']:.4f}")
    print(f"F1 Score (weight):  {metrics['f1_weighted']:.4f}")
    print(f"{'='*60}\n")


def get_metric_summary(metrics):
    """
    Get a summary string of key metrics.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Formatted summary string
    """
    return (f"Acc: {metrics['accuracy']:.4f}, "
            f"Prec: {metrics['precision_macro']:.4f}, "
            f"Rec: {metrics['recall_macro']:.4f}, "
            f"F1: {metrics['f1_macro']:.4f}")

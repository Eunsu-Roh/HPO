"""XGBoost model builder for MNIST classification."""

import xgboost as xgb


def build_xgboost_model(num_classes=10, **params):
    """
    Build an XGBoost classifier with specified parameters.
    
    Args:
        num_classes: Number of output classes (default: 10)
        **params: Model parameters
            
    Returns:
        XGBClassifier instance
    """
    # Default parameters for multi-class classification
    default_params = {
        'objective': 'multi:softmax',
        'num_class': num_classes,
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 100,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'random_state': 42,
        'n_jobs': 10,
        'verbosity': 0,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist'
    }
    
    # Update with provided parameters
    model_params = {**default_params, **params}
    
    # Ensure objective and num_class are set correctly
    model_params['objective'] = 'multi:softmax'
    model_params['num_class'] = num_classes
    
    # Create and return model
    model = xgb.XGBClassifier(**model_params)
    
    return model

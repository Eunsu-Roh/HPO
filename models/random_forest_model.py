"""Random Forest model builder for MNIST classification."""

from sklearn.ensemble import RandomForestClassifier


def build_random_forest_model(**params):
    """
    Build a Random Forest classifier with specified parameters.
    
    Args:
        **params: Model parameters including:
            - n_estimators: Number of trees (default: 100)
            - max_depth: Maximum depth of trees (default: None)
            - min_samples_split: Minimum samples to split (default: 2)
            - min_samples_leaf: Minimum samples per leaf (default: 1)
            - max_features: Number of features to consider (default: 'sqrt')
            - criterion: Split criterion (default: 'gini')
            - bootstrap: Whether to use bootstrap samples (default: True)
            - random_state: Random seed (default: 42)
            - n_jobs: Number of parallel jobs (default: -1)
            
    Returns:
        RandomForestClassifier instance
    """
    # Default parameters
    default_params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'criterion': 'gini',
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 0
    }
    
    # Update with provided parameters
    model_params = {**default_params, **params}
    
    # Create and return model
    model = RandomForestClassifier(**model_params)
    
    return model

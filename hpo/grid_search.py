"""Grid Search hyperparameter optimization."""

import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


class GridSearchOptimizer:
    """Grid Search optimization for sklearn-compatible models."""
    
    def __init__(self, cv=3, n_jobs=4, verbose=2, random_state=42):  # n_jobs 기본값 4로 변경
        """
        Initialize Grid Search optimizer.
        
        Args:
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs (4 for stability on 12-core CPU)
            verbose: Verbosity level
            random_state: Random seed
        """
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.search_time_ = None
        self.grid_search_ = None
    
    def _get_param_grid(self, model_type, config):
        """
        Get parameter grid for specific model type from config.
        
        Args:
            model_type: Type of model ('cnn', 'random_forest', 'xgboost')
            config: Configuration dictionary
            
        Returns:
            Parameter grid for GridSearchCV
        """
        hpo_params = config['hyperparameters'][model_type]['grid_search']
        
        if model_type == 'random_forest':
            return {
                'n_estimators': hpo_params['n_estimators'],
                'max_depth': hpo_params['max_depth'],
                'min_samples_split': hpo_params['min_samples_split']
            }
        elif model_type == 'xgboost':
            return {
                'learning_rate': hpo_params['learning_rate'],
                'max_depth': hpo_params['max_depth'],
                'n_estimators': hpo_params['n_estimators']
            }
        else:
            raise ValueError(f"Grid search not directly supported for {model_type}")
    
    def optimize(self, model_type, X, y, config):
        """
        Perform grid search optimization.
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost')
            X: Training features
            y: Training labels
            config: Configuration dictionary
            
        Returns:
            Dictionary containing best parameters and score
        """
        print(f"\nStarting Grid Search for {model_type}...")
        
        # Get parameter grid
        param_grid = self._get_param_grid(model_type, config)
        
        # Create base model
        if model_type == 'random_forest':
            base_model = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        elif model_type == 'xgboost':
            base_model = xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=10,
                random_state=self.random_state,
                n_jobs=10,
                verbosity=0
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Perform grid search
        start_time = time.time()
        
        # Calculate total combinations
        total_combinations = 1
        for param_values in param_grid.values():
            total_combinations *= len(param_values)
        
        print(f"  🔍 Testing {total_combinations} combinations with {self.cv}-fold CV...")
        print(f"  💻 Using {self.n_jobs} CPU cores for parallel CV")
        print(f"  📊 Grid parameters: {list(param_grid.keys())}")
        print(f"  ⏳ Starting grid search...")
        
        self.grid_search_ = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.cv,
            scoring='accuracy',
            n_jobs=self.n_jobs,
            verbose=3,  # 최대 상세 출력
            refit=True
        )
        
        self.grid_search_.fit(X, y)
        
        self.search_time_ = time.time() - start_time
        self.best_params_ = self.grid_search_.best_params_
        self.best_score_ = self.grid_search_.best_score_
        
        print(f"Grid Search completed in {self.search_time_:.2f}s")
        print(f"Best CV score: {self.best_score_:.4f}")
        print(f"Best parameters: {self.best_params_}")
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'search_time': self.search_time_,
            'cv_results': self.grid_search_.cv_results_
        }
    
    def get_best_model(self):
        """Get the best model from grid search."""
        if self.grid_search_ is None:
            raise ValueError("Must call optimize() before get_best_model()")
        return self.grid_search_.best_estimator_

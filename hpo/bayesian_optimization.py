"""Bayesian Optimization for hyperparameter tuning."""

import time
import numpy as np
from typing import Optional, Dict, Any
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


class BayesianOptimizer:
    """Bayesian Optimization for sklearn-compatible models."""
    
    def __init__(self, n_iter=50, cv=3, n_jobs=10, verbose=2, random_state=42):
        """
        Initialize Bayesian optimizer.
        
        Args:
            n_iter: Number of optimization iterations
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs (10 cores)
            verbose: Verbosity level
            random_state: Random seed
        """
        self.n_iter = n_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.search_time_: Optional[float] = None
        self.bayes_search_: Optional[BayesSearchCV] = None
    
    def _get_search_spaces(self, model_type, config):
        """
        Get search spaces for specific model type from config.
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost')
            config: Configuration dictionary
            
        Returns:
            Search spaces for BayesSearchCV
        """
        hpo_params = config['hyperparameters'][model_type]['bayesian_optimization']
        
        if model_type == 'random_forest':
            return {
                'n_estimators': Integer(hpo_params['n_estimators'][0],
                                       hpo_params['n_estimators'][1]),
                'max_depth': Integer(hpo_params['max_depth'][0],
                                    hpo_params['max_depth'][1]),
                'min_samples_split': Integer(hpo_params['min_samples_split'][0],
                                            hpo_params['min_samples_split'][1]),
                'min_samples_leaf': Integer(hpo_params['min_samples_leaf'][0],
                                           hpo_params['min_samples_leaf'][1])
            }
        elif model_type == 'xgboost':
            return {
                'learning_rate': Real(hpo_params['learning_rate'][0],
                                     hpo_params['learning_rate'][1],
                                     prior='log-uniform'),
                'max_depth': Integer(hpo_params['max_depth'][0],
                                    hpo_params['max_depth'][1]),
                'n_estimators': Integer(hpo_params['n_estimators'][0],
                                       hpo_params['n_estimators'][1]),
                'subsample': Real(hpo_params['subsample'][0],
                                 hpo_params['subsample'][1]),
                'colsample_bytree': Real(hpo_params['colsample_bytree'][0],
                                        hpo_params['colsample_bytree'][1])
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def optimize(self, model_type, X, y, config):
        """
        Perform Bayesian optimization.
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost')
            X: Training features
            y: Training labels
            config: Configuration dictionary
            
        Returns:
            Dictionary containing best parameters and score
        """
        print(f"\nStarting Bayesian Optimization for {model_type}...")
        
        # Get search spaces
        search_spaces = self._get_search_spaces(model_type, config)
        
        # Create base model with limited n_jobs
        if model_type == 'random_forest':
            base_model = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=10
            )
        elif model_type == 'xgboost':
            base_model = xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=10,
                random_state=self.random_state,
                n_jobs=10,
                verbosity=1,
                tree_method='hist'
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Perform Bayesian optimization
        start_time = time.time()
        
        print(f"  🔍 Running {self.n_iter} iterations with {self.cv}-fold CV...")
        print(f"  💻 Using {self.n_jobs} CPU cores for parallel CV")
        print(f"  📊 Search space: {list(search_spaces.keys())}")
        
        self.bayes_search_ = BayesSearchCV(
            estimator=base_model,
            search_spaces=search_spaces,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring='accuracy',
            n_jobs=10,
            verbose=3,
            random_state=self.random_state,
            refit=True
        )
        
        print(f"  ⏳ Starting Bayesian search...")
        self.bayes_search_.fit(X, y)
        
        self.search_time_ = time.time() - start_time
        self.best_params_ = self.bayes_search_.best_params_  # type: ignore[attr-defined]
        self.best_score_ = self.bayes_search_.best_score_  # type: ignore[attr-defined]
        
        print(f"\n  ✅ Bayesian Optimization completed in {self.search_time_:.2f}s ({self.search_time_/60:.1f}m)")
        print(f"  🎯 Best CV score: {self.best_score_:.4f}")
        print(f"  ⚙️  Best parameters: {self.best_params_}")
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'search_time': self.search_time_,
            'cv_results': self.bayes_search_.cv_results_  # type: ignore[attr-defined]
        }
    
    def get_best_model(self):
        """Get the best model from Bayesian optimization."""
        if self.bayes_search_ is None:
            raise ValueError("Must call optimize() before get_best_model()")
        return self.bayes_search_.best_estimator_  # type: ignore

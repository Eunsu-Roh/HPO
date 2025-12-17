"""Random Search hyperparameter optimization."""

import time
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform, randint
import xgboost as xgb


class RandomSearchOptimizer:
    """Random Search optimization for sklearn-compatible models."""
    
    def __init__(self, n_iter=100, cv=3, n_jobs=4, verbose=2, random_state=42):  # n_jobs 기본값 4로 변경
        """
        Initialize Random Search optimizer.
        
        Args:
            n_iter: Number of iterations
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs (4 for stability on 12-core CPU)
            verbose: Verbosity level
            random_state: Random seed
        """
        self.n_iter = n_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.search_time_ = None
        self.random_search_ = None
    
    def _get_param_distributions(self, model_type, config):
        """
        Get parameter distributions for specific model type from config.
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost')
            config: Configuration dictionary
            
        Returns:
            Parameter distributions for RandomizedSearchCV
        """
        hpo_params = config['hyperparameters'][model_type]['random_search']
        
        if model_type == 'random_forest':
            return {
                'n_estimators': randint(hpo_params['n_estimators'][0], 
                                       hpo_params['n_estimators'][1] + 1),
                'max_depth': randint(hpo_params['max_depth'][0], 
                                    hpo_params['max_depth'][1] + 1),
                'min_samples_split': randint(hpo_params['min_samples_split'][0],
                                            hpo_params['min_samples_split'][1] + 1),
                'min_samples_leaf': randint(hpo_params['min_samples_leaf'][0],
                                           hpo_params['min_samples_leaf'][1] + 1),
                'max_features': hpo_params['max_features']
            }
        elif model_type == 'xgboost':
            return {
                'learning_rate': uniform(hpo_params['learning_rate'][0],
                                        hpo_params['learning_rate'][1] - hpo_params['learning_rate'][0]),
                'max_depth': randint(hpo_params['max_depth'][0],
                                    hpo_params['max_depth'][1] + 1),
                'n_estimators': randint(hpo_params['n_estimators'][0],
                                       hpo_params['n_estimators'][1] + 1),
                'subsample': uniform(hpo_params['subsample'][0],
                                    hpo_params['subsample'][1] - hpo_params['subsample'][0]),
                'colsample_bytree': uniform(hpo_params['colsample_bytree'][0],
                                           hpo_params['colsample_bytree'][1] - hpo_params['colsample_bytree'][0]),
                'gamma': uniform(hpo_params['gamma'][0],
                                hpo_params['gamma'][1] - hpo_params['gamma'][0])
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def optimize(self, model_type, X, y, config):
        """
        Perform random search optimization.
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost')
            X: Training features
            y: Training labels
            config: Configuration dictionary
            
        Returns:
            Dictionary containing best parameters and score
        """
        print(f"\nStarting Random Search for {model_type}...")
        
        # Get parameter distributions
        param_distributions = self._get_param_distributions(model_type, config)
        
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
        
        # Perform random search
        start_time = time.time()
        
        print(f"  🔍 Running {self.n_iter} random iterations with {self.cv}-fold CV...")
        print(f"  💻 Using {self.n_jobs} CPU cores for parallel CV")
        print(f"  📊 Search space: {list(param_distributions.keys())}")
        print(f"  ⏳ Starting random search...")
        
        self.random_search_ = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring='accuracy',
            n_jobs=self.n_jobs,
            verbose=3,  # 최대 상세 출력
            random_state=self.random_state,
            refit=True
        )
        
        self.random_search_.fit(X, y)
        
        self.search_time_ = time.time() - start_time
        self.best_params_ = self.random_search_.best_params_
        self.best_score_ = self.random_search_.best_score_
        
        print(f"Random Search completed in {self.search_time_:.2f}s")
        print(f"Best CV score: {self.best_score_:.4f}")
        print(f"Best parameters: {self.best_params_}")
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'search_time': self.search_time_,
            'cv_results': self.random_search_.cv_results_
        }
    
    def get_best_model(self):
        """Get the best model from random search."""
        if self.random_search_ is None:
            raise ValueError("Must call optimize() before get_best_model()")
        return self.random_search_.best_estimator_

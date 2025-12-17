"""Optuna-based hyperparameter optimization."""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


class OptunaOptimizer:
    """Optuna optimization for all model types including deep learning."""
    
    def __init__(self, n_trials=100, timeout=None, pruning=True, 
                 n_jobs=1, random_state=42):
        """
        Initialize Optuna optimizer.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Time budget in seconds (None for no limit)
            pruning: Whether to use pruning for early stopping
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.pruning = pruning
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.search_time_ = None
        self.study_ = None
    
    def _create_objective_rf(self, X, y, config, cv=3):
        """Create objective function for Random Forest."""
        hpo_params = config['hyperparameters']['random_forest']['optuna']
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators',
                                                  hpo_params['n_estimators'][0],
                                                  hpo_params['n_estimators'][1]),
                'max_depth': trial.suggest_int('max_depth',
                                              hpo_params['max_depth'][0],
                                              hpo_params['max_depth'][1]),
                'min_samples_split': trial.suggest_int('min_samples_split',
                                                      hpo_params['min_samples_split'][0],
                                                      hpo_params['min_samples_split'][1]),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf',
                                                     hpo_params['min_samples_leaf'][0],
                                                     hpo_params['min_samples_leaf'][1]),
                'max_features': trial.suggest_categorical('max_features',
                                                         hpo_params['max_features']),
                'random_state': self.random_state,
                'n_jobs': 6  # 12코어의 50%
            }
            
            model = RandomForestClassifier(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=2)  # CV는 2코어만 사용
            return scores.mean()
        
        return objective
    
    def _create_objective_xgb(self, X, y, config, cv=3):
        """Create objective function for XGBoost."""
        hpo_params = config['hyperparameters']['xgboost']['optuna']
        
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate',
                                                    hpo_params['learning_rate'][0],
                                                    hpo_params['learning_rate'][1],
                                                    log=True),
                'max_depth': trial.suggest_int('max_depth',
                                              hpo_params['max_depth'][0],
                                              hpo_params['max_depth'][1]),
                'n_estimators': trial.suggest_int('n_estimators',
                                                  hpo_params['n_estimators'][0],
                                                  hpo_params['n_estimators'][1]),
                'subsample': trial.suggest_float('subsample',
                                                hpo_params['subsample'][0],
                                                hpo_params['subsample'][1]),
                'colsample_bytree': trial.suggest_float('colsample_bytree',
                                                       hpo_params['colsample_bytree'][0],
                                                       hpo_params['colsample_bytree'][1]),
                'gamma': trial.suggest_float('gamma',
                                            hpo_params['gamma'][0],
                                            hpo_params['gamma'][1]),
                'min_child_weight': trial.suggest_int('min_child_weight',
                                                     hpo_params['min_child_weight'][0],
                                                     hpo_params['min_child_weight'][1]),
                'reg_alpha': trial.suggest_float('reg_alpha',
                                                0.001,  # 0 -> 0.001로 수정 (log=True 오류 방지)
                                                hpo_params['reg_alpha'][1],
                                                log=True),
                'reg_lambda': trial.suggest_float('reg_lambda',
                                                 0.001,  # 0 -> 0.001로 수정
                                                 hpo_params['reg_lambda'][1],
                                                 log=True),
                'objective': 'multi:softmax',
                'num_class': 10,
                'random_state': self.random_state,
                'n_jobs': 10,
                'verbosity': 0,
                'tree_method': 'hist'
            }
            
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=3)
            return scores.mean()
        
        return objective
    
    def _create_objective_cnn(self, X_train, y_train, X_val, y_val, config):
        """Create objective function for CNN."""
        hpo_params = config['hyperparameters']['cnn']['optuna']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # GPU 정보 출력 (한 번만)
        print(f"\n🚀 Optuna CNN Optimization - Device: {device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
        
        def objective(trial):
            # Architecture parameters
            num_conv_layers = trial.suggest_int('num_conv_layers',
                                               hpo_params['num_conv_layers'][0],
                                               hpo_params['num_conv_layers'][1])
            conv_filters = []
            for i in range(num_conv_layers):
                filters = trial.suggest_categorical(f'conv_filters_{i}',
                                                   hpo_params['dense_units'])
                conv_filters.append(filters)
            
            dense_units = trial.suggest_categorical('dense_units',
                                                   hpo_params['dense_units'])
            dropout_rate = trial.suggest_float('dropout_rate',
                                              hpo_params['dropout_rate'][0],
                                              hpo_params['dropout_rate'][1])
            
            # Training parameters
            learning_rate = trial.suggest_float('learning_rate',
                                               hpo_params['learning_rate'][0],
                                               hpo_params['learning_rate'][1],
                                               log=True)
            batch_size = trial.suggest_categorical('batch_size',
                                                  hpo_params['batch_size'])
            epochs = trial.suggest_int('epochs',
                                      hpo_params['epochs'][0],
                                      hpo_params['epochs'][1])
            
            # Import model
            from models.cnn_model import CNNModel
            
            # Build model
            model = CNNModel(
                num_conv_layers=num_conv_layers,
                conv_filters=conv_filters,
                dense_units=dense_units,
                dropout_rate=dropout_rate,
                num_classes=10
            ).to(device)
            
            # Optimizer and loss
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Prepare data
            X_train_tensor = torch.FloatTensor(X_train).view(-1, 1, 28, 28)
            y_train_tensor = torch.LongTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val).view(-1, 1, 28, 28)
            y_val_tensor = torch.LongTensor(y_val)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                    num_workers=0, pin_memory=False)  # Windows 안정성
            
            # Train with pruning
            best_val_acc = 0
            for epoch in range(epochs):
                model.train()
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    X_val_device = X_val_tensor.to(device)
                    y_val_device = y_val_tensor.to(device)
                    
                    val_outputs = model(X_val_device)
                    val_loss = criterion(val_outputs, y_val_device)
                    val_preds = torch.argmax(val_outputs, dim=1)
                    val_acc = (val_preds == y_val_device).float().mean().item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                # Report intermediate value
                trial.report(val_acc, epoch)
                
                # Prune if necessary
                if self.pruning and trial.should_prune():
                    raise optuna.TrialPruned()
            
            return best_val_acc
        
        return objective
    
    def optimize(self, model_type, X, y, config, X_val=None, y_val=None):
        """
        Perform Optuna optimization.
        
        Args:
            model_type: Type of model ('cnn', 'random_forest', 'xgboost')
            X: Training features
            y: Training labels
            config: Configuration dictionary
            X_val: Validation features (for CNN)
            y_val: Validation labels (for CNN)
            
        Returns:
            Dictionary containing best parameters and score
        """
        print(f"\nStarting Optuna Optimization for {model_type}...")
        
        # Create study
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner() if self.pruning else None
        
        self.study_ = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Create objective function
        if model_type == 'cnn':
            if X_val is None or y_val is None:
                raise ValueError("X_val and y_val required for CNN optimization")
            objective = self._create_objective_cnn(X, y, X_val, y_val, config)
        elif model_type == 'random_forest':
            cv = config['hpo_methods']['optuna'].get('cv_folds', 3)
            objective = self._create_objective_rf(X, y, config, cv=cv)
        elif model_type == 'xgboost':
            cv = config['hpo_methods']['optuna'].get('cv_folds', 3)
            objective = self._create_objective_xgb(X, y, config, cv=cv)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Optimize
        start_time = time.time()
        
        print(f"  🔍 Running {self.n_trials} trials...")
        if model_type == 'cnn':
            print(f"  🎮 Using GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
        else:
            print(f"  💻 Using 6 CPU cores for XGBoost, 2 cores for CV")
        
        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        print(f"  ⏳ Starting Optuna TPE search...")
        self.study_.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
            callbacks=[lambda study, trial: print(f"    Trial {trial.number+1}/{self.n_trials}: score={trial.value:.4f} {'(pruned)' if trial.state == optuna.trial.TrialState.PRUNED else ''}")]
        )
        
        self.search_time_ = time.time() - start_time
        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value
        
        pruned = len([t for t in self.study_.trials if t.state == optuna.trial.TrialState.PRUNED])
        
        print(f"\n  ✅ Optuna Optimization completed in {self.search_time_:.2f}s ({self.search_time_/60:.1f}m)")
        print(f"  🎯 Best score: {self.best_score_:.4f}")
        print(f"  ⚙️  Best parameters: {self.best_params_}")
        print(f"  📊 Total trials: {len(self.study_.trials)} ({pruned} pruned)")
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'search_time': self.search_time_,
            'study': self.study_
        }
    
    def get_study(self):
        """Get the Optuna study object."""
        return self.study_

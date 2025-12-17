"""Experiment runner for executing all HPO combinations."""

import time
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data import MNISTDataLoader
from models import build_cnn_model, build_random_forest_model, build_xgboost_model
from hpo import GridSearchOptimizer, RandomSearchOptimizer, BayesianOptimizer, OptunaOptimizer
from evaluation.metrics import evaluate_model
from .experiment_tracker import ExperimentTracker


class ExperimentRunner:
    """Run all model-HPO combinations in optimized order."""
    
    def __init__(self, config, data_loader):
        """
        Initialize experiment runner.
        
        Args:
            config: Configuration dictionary
            data_loader: MNISTDataLoader instance
        """
        self.config = config
        self.data_loader = data_loader
        self.tracker = ExperimentTracker(
            experiment_name=config['experiment']['name'],
            save_dir=config['experiment']['save_dir'],
            log_dir=config['logging']['log_dir']
        )
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config['experiment']['save_dir']) / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.tracker.logger.info("ExperimentRunner initialized")
    
    def _build_execution_plan(self):
        """
        Build execution plan based on config order.
        
        Returns:
            List of (model_name, hpo_method) tuples
        """
        execution_plan = []
        for item in self.config['execution_order']:
            model = item['model']
            for hpo_method in item['hpo_methods']:
                execution_plan.append((model, hpo_method))
        
        self.tracker.logger.info(f"Execution plan: {len(execution_plan)} combinations")
        for i, (model, hpo) in enumerate(execution_plan, 1):
            self.tracker.logger.info(f"  {i}. {model} - {hpo}")
        
        return execution_plan
    
    def _run_ml_experiment(self, model_name, hpo_method):
        """
        Run experiment for ML models (RandomForest, XGBoost).
        
        Args:
            model_name: Name of the model
            hpo_method: HPO method to use
            
        Returns:
            Dictionary with results
        """
        self.tracker.logger.info(f"_run_ml_experiment called: {model_name}, {hpo_method}")
        
        # Get data
        self.tracker.logger.info("Loading data...")
        X_train, y_train, X_val, y_val, X_test, y_test = self.data_loader.get_data_for_ml()
        X_cv, y_cv = self.data_loader.get_combined_train_val_for_cv(for_cnn=False)
        self.tracker.logger.info(f"Data loaded: X_cv shape={X_cv.shape}, y_cv shape={y_cv.shape}")
        
        # Initialize optimizer
        if hpo_method == 'grid_search':
            optimizer = GridSearchOptimizer(
                cv=self.config['hpo_methods']['grid_search']['cv_folds'],
                n_jobs=self.config['hpo_methods']['grid_search']['n_jobs'],
                verbose=self.config['hpo_methods']['grid_search']['verbose'],
                random_state=self.config['experiment']['seed']
            )
        elif hpo_method == 'random_search':
            optimizer = RandomSearchOptimizer(
                n_iter=self.config['hpo_methods']['random_search']['n_iter'],
                cv=self.config['hpo_methods']['random_search']['cv_folds'],
                n_jobs=self.config['hpo_methods']['random_search']['n_jobs'],
                verbose=self.config['hpo_methods']['random_search']['verbose'],
                random_state=self.config['experiment']['seed']
            )
        elif hpo_method == 'bayesian_optimization':
            optimizer = BayesianOptimizer(
                n_iter=self.config['hpo_methods']['bayesian_optimization']['n_iter'],
                cv=self.config['hpo_methods']['bayesian_optimization']['cv_folds'],
                n_jobs=self.config['hpo_methods']['bayesian_optimization']['n_jobs'],
                verbose=self.config['hpo_methods']['bayesian_optimization']['verbose'],
                random_state=self.config['experiment']['seed']
            )
        elif hpo_method == 'optuna':
            optimizer = OptunaOptimizer(
                n_trials=self.config['hpo_methods']['optuna']['n_trials'],
                timeout=self.config['hpo_methods']['optuna']['timeout_per_trial'],
                pruning=self.config['hpo_methods']['optuna']['pruning'],
                n_jobs=self.config['hpo_methods']['optuna']['n_jobs'],
                random_state=self.config['experiment']['seed']
            )
        else:
            raise ValueError(f"Unknown HPO method: {hpo_method}")
        
        # Optimize
        hpo_results = optimizer.optimize(model_name, X_cv, y_cv, self.config)
        
        # Build best model with found parameters
        if model_name == 'random_forest':
            best_model = build_random_forest_model(**hpo_results['best_params'])
            model_type = 'sklearn'
        else:  # xgboost
            best_model = build_xgboost_model(**hpo_results['best_params'])
            model_type = 'xgboost'
        
        # Train on full train+val data
        best_model.fit(X_cv, y_cv)
        
        # Evaluate on test set
        test_metrics, cm, report = evaluate_model(best_model, X_test, y_test, model_type=model_type)
        
        return {
            'best_params': hpo_results['best_params'],
            'best_cv_score': hpo_results['best_score'],
            'search_time': hpo_results['search_time'],
            'test_metrics': test_metrics,
            'confusion_matrix': cm,
            'model': best_model,
            'hpo_results': hpo_results
        }
    
    def _run_cnn_experiment(self, hpo_method):
        """
        Run experiment for CNN model.
        
        Args:
            hpo_method: HPO method to use
            
        Returns:
            Dictionary with results
        """
        # GPU 메모리 초기화 (이전 실험의 메모리 정리)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Get data
        X_train, y_train, X_val, y_val, X_test, y_test = self.data_loader.get_data_for_cnn()
        
        # GPU 설정 및 정보 출력
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tracker.logger.info(f"🚀 CNN Training Device: {device}")
        if torch.cuda.is_available():
            self.tracker.logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            self.tracker.logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.tracker.logger.warning("   ⚠️  GPU not available, using CPU")
        
        if hpo_method == 'optuna':
            # Use Optuna for CNN optimization
            optimizer = OptunaOptimizer(
                n_trials=self.config['hpo_methods']['optuna']['n_trials'],
                timeout=self.config['hpo_methods']['optuna']['timeout_per_trial'],
                pruning=self.config['hpo_methods']['optuna']['pruning'],
                n_jobs=self.config['hpo_methods']['optuna']['n_jobs'],
                random_state=self.config['experiment']['seed']
            )
            
            hpo_results = optimizer.optimize('cnn', X_train, y_train, self.config, X_val, y_val)
            
            # Build and train best model
            best_params = hpo_results['best_params']
            
            # Extract conv_filters from individual parameters
            num_conv_layers = best_params['num_conv_layers']
            conv_filters = [best_params[f'conv_filters_{i}'] for i in range(num_conv_layers)]
            
            model_params = {
                'num_conv_layers': num_conv_layers,
                'conv_filters': conv_filters,
                'dense_units': best_params['dense_units'],
                'dropout_rate': best_params['dropout_rate']
            }
            
            # Build model
            from models.cnn_model import CNNModel
            best_model = CNNModel(**model_params, num_classes=10).to(device)
            learning_rate = best_params['learning_rate']
            batch_size = best_params['batch_size']
            epochs = best_params['epochs']
            
            # Train on train+val
            X_train_full = np.concatenate([X_train, X_val], axis=0)
            y_train_full = np.concatenate([y_train, y_val], axis=0)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_train_full).view(-1, 1, 28, 28)
            y_tensor = torch.LongTensor(y_train_full)
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)  # Windows 안정성
            
            # Train
            optimizer_torch = optim.Adam(best_model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            best_model.train()
            for epoch in range(epochs):
                for batch_x, batch_y in dataloader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    optimizer_torch.zero_grad()
                    outputs = best_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer_torch.step()
            
        else:
            # For other methods, use simplified grid/random search for CNN
            self.tracker.logger.warning(f"CNN + {hpo_method}: Using simplified parameter search")
            
            # Sample a few configurations
            configs_to_try = self._get_cnn_sample_configs(hpo_method)
            
            best_val_acc = 0
            best_config = None
            best_model = None
            search_start = time.time()
            
            for config_params in tqdm(configs_to_try, desc=f"CNN {hpo_method}"):
                # Build model
                from models.cnn_model import CNNModel
                model = CNNModel(**config_params['model_params'], num_classes=10).to(device)
                
                # Prepare data
                X_train_tensor = torch.FloatTensor(X_train).view(-1, 1, 28, 28)
                y_train_tensor = torch.LongTensor(y_train)
                X_val_tensor = torch.FloatTensor(X_val).view(-1, 1, 28, 28)
                y_val_tensor = torch.LongTensor(y_val)
                
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=config_params['batch_size'], shuffle=True,
                                        num_workers=0, pin_memory=False)  # Windows 안정성
                
                # Train
                optimizer_torch = optim.Adam(model.parameters(), lr=config_params['learning_rate'])
                criterion = nn.CrossEntropyLoss()
                
                val_accs = []
                for epoch in range(config_params['epochs']):
                    model.train()
                    for batch_x, batch_y in train_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        
                        optimizer_torch.zero_grad()
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer_torch.step()
                    
                    # Validation
                    model.eval()
                    with torch.no_grad():
                        X_val_device = X_val_tensor.to(device)
                        y_val_device = y_val_tensor.to(device)
                        val_outputs = model(X_val_device)
                        val_preds = torch.argmax(val_outputs, dim=1)
                        val_acc = (val_preds == y_val_device).float().mean().item()
                        val_accs.append(val_acc)
                
                max_val_acc = max(val_accs)
                if max_val_acc > best_val_acc:
                    best_val_acc = max_val_acc
                    best_config = config_params
                    best_model = model
            
            search_time = time.time() - search_start
            
            hpo_results = {
                'best_params': best_config,
                'best_score': best_val_acc,
                'search_time': search_time
            }
        
        # Evaluate on test set
        test_metrics, cm, report = evaluate_model(best_model, X_test, y_test, model_type='pytorch')
        
        return {
            'best_params': hpo_results['best_params'],
            'best_cv_score': hpo_results['best_score'],
            'search_time': hpo_results['search_time'],
            'test_metrics': test_metrics,
            'confusion_matrix': cm,
            'model': best_model,
            'hpo_results': hpo_results
        }
    
    def _get_cnn_sample_configs(self, hpo_method):
        """Get sample CNN configurations for non-Optuna methods."""
        hpo_params = self.config['hyperparameters']['cnn'][hpo_method]
        
        if hpo_method == 'grid_search':
            # Small grid for demonstration
            configs = []
            for lr in hpo_params['learning_rate']:
                for epochs in hpo_params['epochs']:
                    for batch_size in hpo_params['batch_size']:
                        for num_layers in hpo_params['num_conv_layers']:
                            configs.append({
                                'model_params': {
                                    'num_conv_layers': num_layers,
                                    'dropout_rate': 0.3
                                },
                                'learning_rate': lr,
                                'batch_size': batch_size,
                                'epochs': epochs
                            })
            return configs[:10]  # Limit to 10 configs
        else:
            # Random sample for random_search and bayesian
            import random
            random.seed(self.config['experiment']['seed'])
            
            configs = []
            n_samples = 10 if hpo_method == 'bayesian_optimization' else 15
            
            for _ in range(n_samples):
                if isinstance(hpo_params['learning_rate'][0], list):
                    lr_min, lr_max = hpo_params['learning_rate']
                    lr = 10 ** random.uniform(np.log10(lr_min), np.log10(lr_max))
                else:
                    lr = random.choice(hpo_params['learning_rate'])
                
                configs.append({
                    'model_params': {
                        'num_conv_layers': random.choice(hpo_params['num_conv_layers']),
                        'dropout_rate': random.uniform(hpo_params['dropout_rate'][0], hpo_params['dropout_rate'][1])
                    },
                    'learning_rate': lr,
                    'batch_size': random.choice(hpo_params['batch_size']),
                    'epochs': random.randint(hpo_params['epochs'][0], hpo_params['epochs'][1])
                })
            
            return configs
    
    def _save_model_checkpoint(self, model, model_name, hpo_method):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{model_name}_{hpo_method}.pkl"
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(model, f)
            self.tracker.logger.info(f"Model checkpoint saved: {checkpoint_path}")
        except Exception as e:
            self.tracker.logger.error(f"Error saving checkpoint: {e}")
    
    def run_all_experiments(self):
        """Run all experiments in the execution plan, skipping already completed ones."""
        execution_plan = self._build_execution_plan()
        
        # Check which experiments are already completed
        completed_experiments = set()
        if self.tracker.results:
            for result in self.tracker.results:
                completed_experiments.add((result['model'], result['hpo_method']))
            print(f"\n✓ Found {len(completed_experiments)} already completed experiments")
            for model, hpo in sorted(completed_experiments):
                print(f"  {model} - {hpo}")
        
        # Filter execution plan to skip completed experiments
        remaining_plan = [
            (model, hpo) for model, hpo in execution_plan 
            if (model, hpo) not in completed_experiments
        ]
        
        if not remaining_plan:
            print("\n✓ All experiments already completed!")
            print(f"Total: {len(execution_plan)} experiments")
            self.tracker.print_summary()
            return self.tracker
        
        print(f"\n{'='*80}")
        print(f"Starting {len(remaining_plan)} experiments ({len(completed_experiments)} already completed)")
        print(f"Total progress: {len(completed_experiments)}/{len(execution_plan)} experiments done")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        experiment_times = []  # Track individual experiment times for ETA
        
        # Progress bar for overall experiments
        with tqdm(total=len(remaining_plan), desc="Overall Progress", 
                  unit="exp", ncols=100, position=0, leave=True) as pbar:
            
            for i, (model_name, hpo_method) in enumerate(remaining_plan, 1):
                overall_idx = len(completed_experiments) + i
                exp_start = time.time()
                
                # Update progress bar description
                pbar.set_description(f"[{overall_idx}/{len(execution_plan)}] {model_name}-{hpo_method}")
                
                print(f"\n{'─'*80}")
                print(f"📊 Experiment {overall_idx}/{len(execution_plan)}: {model_name} - {hpo_method}")
                
                # Calculate and show ETA
                if experiment_times:
                    avg_time = sum(experiment_times) / len(experiment_times)
                    remaining_exp = len(remaining_plan) - i + 1
                    eta_seconds = avg_time * remaining_exp
                    eta_hours = eta_seconds / 3600
                    elapsed_hours = (time.time() - start_time) / 3600
                    total_progress_pct = (len(completed_experiments) + i - 1) / len(execution_plan) * 100
                    
                    print(f"⏱️  Progress: {total_progress_pct:.1f}% | "
                          f"Elapsed: {elapsed_hours:.2f}h | "
                          f"ETA: {eta_hours:.2f}h")
                print(f"{'─'*80}")
                
                try:
                    # Log start
                    self.tracker.logger.info(f"Starting experiment: {model_name} - {hpo_method}")
                    
                    # Run experiment
                    if model_name == 'cnn':
                        self.tracker.logger.info("Running CNN experiment...")
                        results = self._run_cnn_experiment(hpo_method)
                    else:
                        self.tracker.logger.info(f"Running ML experiment for {model_name}...")
                        results = self._run_ml_experiment(model_name, hpo_method)
                    
                    self.tracker.logger.info("Experiment completed, logging results...")
                    
                    # Log results
                    self.tracker.log_experiment(
                        model_name=model_name,
                        hpo_method=hpo_method,
                        best_params=results['best_params'],
                        best_score=results['best_cv_score'],
                        search_time=results['search_time'],
                        test_metrics=results['test_metrics'],
                        confusion_matrix=results['confusion_matrix']
                    )
                    
                    # Save model checkpoint
                    if self.config['experiment']['save_all_checkpoints']:
                        self._save_model_checkpoint(results['model'], model_name, hpo_method)
                    
                    exp_time = time.time() - exp_start
                    experiment_times.append(exp_time)
                    
                    print(f"\n✅ Completed in {exp_time:.1f}s ({exp_time/60:.1f}m)")
                    print(f"   Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
                    print(f"   CV Score: {results['best_cv_score']:.4f}")
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Update postfix with current stats
                    if experiment_times:
                        avg_time_per_exp = sum(experiment_times) / len(experiment_times)
                        pbar.set_postfix({
                            'avg_time': f'{avg_time_per_exp/60:.1f}m',
                            'last_acc': f'{results["test_metrics"]["accuracy"]:.3f}'
                        })
                    
                except Exception as e:
                    self.tracker.logger.error(f"Error in {model_name} - {hpo_method}: {e}")
                    print(f"\n❌ Failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Still update progress bar
                    pbar.update(1)
                    continue
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"🎉 ALL EXPERIMENTS COMPLETED!")
        print(f"{'='*80}")
        print(f"⏱️  Total Time: {total_time:.1f}s ({total_time/60:.1f}m / {total_time/3600:.2f}h)")
        print(f"📊 Completed: {len(execution_plan)}/{len(execution_plan)} experiments")
        if experiment_times:
            print(f"⚡ Avg Time per Experiment: {sum(experiment_times)/len(experiment_times)/60:.1f}m")
        print(f"{'='*80}\n")
        
        # Print summary
        self.tracker.print_summary()
        
        return self.tracker

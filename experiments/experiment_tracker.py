"""Experiment tracker for logging and saving results."""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd


class ExperimentTracker:
    """Track and save experiment results incrementally."""
    
    def __init__(self, experiment_name, save_dir="results", log_dir="results/logs"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            save_dir: Directory to save results
            log_dir: Directory to save logs
        """
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results list
        self.results: List[Dict[str, Any]] = []
        
        # Setup logging
        self._setup_logging()
        
        # File paths
        self.json_path = self.save_dir / f"{experiment_name}_results.json"
        self.csv_path = self.save_dir / f"{experiment_name}_results.csv"
        
        # Load existing results if available
        self._load_existing_results()
        
        self.logger.info(f"ExperimentTracker initialized: {experiment_name}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"experiment_{timestamp}.log"
        
        # Create logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _load_existing_results(self):
        """Load existing results if available."""
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)
                self.logger.info(f"Loaded {len(self.results)} existing results")
            except Exception as e:
                self.logger.warning(f"Could not load existing results: {e}")
                self.results = []
    
    def log_experiment(self, model_name, hpo_method, best_params, best_score,
                      search_time, test_metrics, confusion_matrix=None):
        """
        Log an experiment result.
        
        Args:
            model_name: Name of the model
            hpo_method: HPO method used
            best_params: Best hyperparameters found
            best_score: Best cross-validation score
            search_time: Time taken for HPO search
            test_metrics: Dictionary of test metrics
            confusion_matrix: Confusion matrix (optional)
        """
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': model_name,
            'hpo_method': hpo_method,
            'best_params': best_params,
            'best_cv_score': float(best_score),
            'search_time': float(search_time),
            'test_accuracy': float(test_metrics.get('accuracy', 0)),
            'test_precision_macro': float(test_metrics.get('precision_macro', 0)),
            'test_precision_weighted': float(test_metrics.get('precision_weighted', 0)),
            'test_recall_macro': float(test_metrics.get('recall_macro', 0)),
            'test_recall_weighted': float(test_metrics.get('recall_weighted', 0)),
            'test_f1_macro': float(test_metrics.get('f1_macro', 0)),
            'test_f1_weighted': float(test_metrics.get('f1_weighted', 0)),
        }
        
        # Add per-class accuracies
        for i in range(10):
            key = f'class_{i}_accuracy'
            if key in test_metrics:
                result[key] = float(test_metrics[key])
        
        # Add confusion matrix if provided
        if confusion_matrix is not None:
            result['confusion_matrix'] = confusion_matrix.tolist()
        
        self.results.append(result)
        
        # Log to file
        self.logger.info(f"Logged: {model_name} - {hpo_method}")
        self.logger.info(f"  CV Score: {best_score:.4f}, Test Acc: {test_metrics.get('accuracy', 0):.4f}, Time: {search_time:.2f}s")
        
        # Save immediately
        self.save_results()
    
    def save_results(self):
        """Save results to JSON and CSV files."""
        try:
            # Save as JSON
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            # Save as CSV (without confusion matrix for readability)
            results_for_csv = []
            for result in self.results:
                result_copy = result.copy()
                result_copy.pop('confusion_matrix', None)
                # Convert best_params dict to string for CSV
                result_copy['best_params'] = str(result_copy['best_params'])
                results_for_csv.append(result_copy)
            
            df = pd.DataFrame(results_for_csv)
            df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"Results saved: {len(self.results)} experiments")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def get_summary(self):
        """
        Get summary statistics of all experiments.
        
        Returns:
            DataFrame with summary statistics
        """
        if not self.results:
            return None
        
        df = pd.DataFrame(self.results)
        
        # Summary by HPO method
        hpo_summary = df.groupby('hpo_method').agg({
            'test_accuracy': ['mean', 'std', 'max', 'min'],
            'search_time': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        # Summary by model
        model_summary = df.groupby('model').agg({
            'test_accuracy': ['mean', 'std', 'max', 'min'],
            'search_time': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        return {
            'hpo_summary': hpo_summary,
            'model_summary': model_summary,
            'overall': {
                'total_experiments': len(self.results),
                'avg_accuracy': df['test_accuracy'].mean(),
                'best_accuracy': df['test_accuracy'].max(),
                'total_time': df['search_time'].sum()
            }
        }
    
    def get_best_result(self) -> Optional[Dict[str, Any]]:
        """Get the best result based on test accuracy."""
        if not self.results:
            return None
        
        df = pd.DataFrame(self.results)
        best_idx = int(df['test_accuracy'].idxmax())  # idxmax()는 인덱스를 반환, int로 변환
        return self.results[best_idx]
    
    def get_results_dataframe(self) -> Optional[pd.DataFrame]:
        """Get results as a pandas DataFrame."""
        if not self.results:
            return None
        return pd.DataFrame(self.results)
    
    def print_summary(self):
        """Print a formatted summary of results."""
        summary = self.get_summary()
        if summary is None:
            print("No results to summarize.")
            return
        
        print("\n" + "="*80)
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print("="*80)
        
        print(f"\nTotal Experiments: {summary['overall']['total_experiments']}")
        print(f"Average Accuracy: {summary['overall']['avg_accuracy']:.4f}")
        print(f"Best Accuracy: {summary['overall']['best_accuracy']:.4f}")
        print(f"Total Time: {summary['overall']['total_time']:.2f}s ({summary['overall']['total_time']/3600:.2f}h)")
        
        print("\n" + "-"*80)
        print("SUMMARY BY HPO METHOD")
        print("-"*80)
        print(summary['hpo_summary'])
        
        print("\n" + "-"*80)
        print("SUMMARY BY MODEL")
        print("-"*80)
        print(summary['model_summary'])
        
        best = self.get_best_result()
        if best is not None:  # best가 None이 아닐 때만 출력
            print("\n" + "-"*80)
            print("BEST RESULT")
            print("-"*80)
            print(f"Model: {best['model']}")
            print(f"HPO Method: {best['hpo_method']}")
            print(f"Test Accuracy: {best['test_accuracy']:.4f}")
            print(f"Best Params: {best['best_params']}")
        print("="*80 + "\n")

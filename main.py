"""Main script to run all HPO experiments."""

import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime

# CPU 사용량 제한 설정 (프로그램 시작 시)
os.environ['OMP_NUM_THREADS'] = '10'  # OpenMP 스레드 제한
os.environ['MKL_NUM_THREADS'] = '10'  # Intel MKL 스레드 제한
os.environ['OPENBLAS_NUM_THREADS'] = '10'  # OpenBLAS 스레드 제한
os.environ['NUMEXPR_NUM_THREADS'] = '10'  # NumExpr 스레드 제한

import torch
torch.set_num_threads(10)  # PyTorch CPU 스레드 제한

from data import MNISTDataLoader
from experiments import ExperimentRunner
from evaluation.visualizer import ResultVisualizer


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_summary_report(tracker, save_dir):
    """
    Save a summary report to text file.
    
    Args:
        tracker: ExperimentTracker instance
        save_dir: Directory to save report
    """
    save_path = Path(save_dir) / 'summary_report.txt'
    
    summary = tracker.get_summary()
    best = tracker.get_best_result()
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MNIST HYPERPARAMETER OPTIMIZATION EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Overall Statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Experiments:     {summary['overall']['total_experiments']}\n")
        f.write(f"Average Accuracy:      {summary['overall']['avg_accuracy']:.4f}\n")
        f.write(f"Best Accuracy:         {summary['overall']['best_accuracy']:.4f}\n")
        f.write(f"Total Time:            {summary['overall']['total_time']:.2f}s ({summary['overall']['total_time']/3600:.2f}h)\n")
        f.write("\n")
        
        # Best Result
        f.write("BEST RESULT\n")
        f.write("-"*80 + "\n")
        f.write(f"Model:                 {best['model']}\n")
        f.write(f"HPO Method:            {best['hpo_method']}\n")
        f.write(f"Test Accuracy:         {best['test_accuracy']:.4f}\n")
        f.write(f"CV Score:              {best['best_cv_score']:.4f}\n")
        f.write(f"Search Time:           {best['search_time']:.2f}s\n")
        f.write(f"F1 Score (macro):      {best['test_f1_macro']:.4f}\n")
        f.write(f"Precision (macro):     {best['test_precision_macro']:.4f}\n")
        f.write(f"Recall (macro):        {best['test_recall_macro']:.4f}\n")
        f.write(f"Best Parameters:\n")
        for param, value in best['best_params'].items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")
        
        # HPO Method Summary
        f.write("SUMMARY BY HPO METHOD\n")
        f.write("-"*80 + "\n")
        hpo_summary = summary['hpo_summary']
        f.write(hpo_summary.to_string())
        f.write("\n\n")
        
        # Model Summary
        f.write("SUMMARY BY MODEL\n")
        f.write("-"*80 + "\n")
        model_summary = summary['model_summary']
        f.write(model_summary.to_string())
        f.write("\n\n")
        
        # Detailed Results
        f.write("DETAILED RESULTS\n")
        f.write("-"*80 + "\n")
        results_df = tracker.get_results_dataframe()
        results_df_display = results_df[['model', 'hpo_method', 'test_accuracy', 
                                         'best_cv_score', 'search_time']].copy()
        results_df_display = results_df_display.sort_values('test_accuracy', ascending=False)
        f.write(results_df_display.to_string(index=False))
        f.write("\n\n")
        
        # Rankings
        f.write("RANKINGS\n")
        f.write("-"*80 + "\n")
        f.write("\nTop 5 by Test Accuracy:\n")
        top_acc = results_df.nlargest(5, 'test_accuracy')[['model', 'hpo_method', 'test_accuracy', 'search_time']]
        for i, (idx, row) in enumerate(top_acc.iterrows(), 1):
            f.write(f"  {i}. {row['model']:15s} - {row['hpo_method']:25s} "
                   f"Acc: {row['test_accuracy']:.4f}  Time: {row['search_time']:>8.2f}s\n")
        
        f.write("\nTop 5 by Efficiency (Accuracy / log(Time)):\n")
        results_df['efficiency'] = results_df['test_accuracy'] / (results_df['search_time'].apply(lambda x: max(x, 1)).apply(lambda x: x**0.5))
        top_eff = results_df.nlargest(5, 'efficiency')[['model', 'hpo_method', 'test_accuracy', 'search_time', 'efficiency']]
        for i, (idx, row) in enumerate(top_eff.iterrows(), 1):
            f.write(f"  {i}. {row['model']:15s} - {row['hpo_method']:25s} "
                   f"Acc: {row['test_accuracy']:.4f}  Time: {row['search_time']:>8.2f}s  Eff: {row['efficiency']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\nSummary report saved: {save_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run MNIST HPO experiments')
    parser.add_argument('--config', type=str, default='config/hpo_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--skip-experiments', action='store_true',
                       help='Skip experiments and only generate visualizations')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MNIST Hyperparameter Optimization Experiment")
    print("="*80)
    
    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"\n🚀 GPU AVAILABLE!")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   PyTorch Version: {torch.__version__}")
    else:
        print(f"\n⚠️  GPU NOT AVAILABLE - Using CPU only")
    print("")
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Random seed: {config['experiment']['seed']}\n")
    
    if not args.skip_experiments:
        # Load data
        print("Loading MNIST dataset...")
        data_loader = MNISTDataLoader(
            train_ratio=config['data']['train_ratio'],
            val_ratio=config['data']['val_ratio'],
            test_ratio=config['data']['test_ratio'],
            normalize=config['data']['normalize'],
            random_state=config['experiment']['seed']
        )
        print(f"Dataset info: {data_loader.get_info()}\n")
        
        # Run experiments
        runner = ExperimentRunner(config, data_loader)
        tracker = runner.run_all_experiments()
        
        # Save summary report
        save_summary_report(tracker, config['experiment']['save_dir'])
    else:
        # Load existing results
        print("Loading existing results...")
        from experiments import ExperimentTracker
        tracker = ExperimentTracker(
            experiment_name=config['experiment']['name'],
            save_dir=config['experiment']['save_dir']
        )
    
    # Generate visualizations
    results_df = tracker.get_results_dataframe()
    
    if results_df is not None and len(results_df) > 0:
        print("\nGenerating visualizations...")
        visualizer = ResultVisualizer(
            results_df,
            save_dir=config['experiment']['save_dir'] + '/plots'
        )
        
        # Try to get Optuna studies if available
        # Note: In a real scenario, you'd need to save and load these
        visualizer.create_all_visualizations(optuna_studies=None)
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE!")
        print("="*80)
        print(f"\nResults saved in: {config['experiment']['save_dir']}")
        print(f"  - experiment_results.json")
        print(f"  - experiment_results.csv")
        print(f"  - summary_report.txt")
        print(f"  - plots/")
        print(f"  - checkpoints/")
        print(f"  - logs/\n")
    else:
        print("\nNo results found. Run experiments first.")


if __name__ == '__main__':
    main()

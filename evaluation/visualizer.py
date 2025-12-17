"""Visualization utilities for experiment results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ResultVisualizer:
    """Visualize experiment results."""
    
    def __init__(self, results_df, save_dir="results/plots"):
        """
        Initialize visualizer.
        
        Args:
            results_df: DataFrame with experiment results
            save_dir: Directory to save plots
        """
        self.results = results_df
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_accuracy_vs_time(self, save=True):
        """Plot accuracy vs search time scatter plot."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color map for HPO methods
        hpo_methods = self.results['hpo_method'].unique()
        colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(hpo_methods)))  # get_cmap 사용
        color_map = dict(zip(hpo_methods, colors))
        
        # Marker map for models
        models = self.results['model'].unique()
        markers = ['o', 's', '^']
        marker_map = dict(zip(models, markers))
        
        # Plot each combination
        for hpo in hpo_methods:
            for model in models:
                mask = (self.results['hpo_method'] == hpo) & (self.results['model'] == model)
                data = self.results[mask]
                
                if len(data) > 0:
                    ax.scatter(
                        data['search_time'],
                        data['test_accuracy'],
                        c=[color_map[hpo]],
                        marker=marker_map[model],
                        s=200,
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=1.5,
                        label=f'{model} - {hpo}'
                    )
        
        ax.set_xlabel('Search Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Hyperparameter Optimization Performance: Accuracy vs. Search Time', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / 'accuracy_vs_time.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        return fig
    
    def plot_model_hpo_comparison(self, save=True):
        """Plot grouped bar charts for model and HPO method comparisons."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Accuracy by Model and HPO Method
        pivot_acc = self.results.pivot(index='model', columns='hpo_method', values='test_accuracy')
        pivot_acc.plot(kind='bar', ax=axes[0, 0], width=0.8, edgecolor='black', linewidth=0.7)
        axes[0, 0].set_title('Test Accuracy Comparison by Model and HPO Method', fontweight='bold', fontsize=12)
        axes[0, 0].set_ylabel('Test Accuracy', fontweight='bold')
        axes[0, 0].set_xlabel('Model Type', fontweight='bold')
        axes[0, 0].legend(title='HPO Method', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].tick_params(axis='x', rotation=0)
        axes[0, 0].set_ylim(0.90, 1.00)  # 90%~100% 범위로 확대
        
        # 2. Search Time by Model and HPO Method
        pivot_time = self.results.pivot(index='model', columns='hpo_method', values='search_time')
        pivot_time.plot(kind='bar', ax=axes[0, 1], width=0.8, edgecolor='black', linewidth=0.7)
        axes[0, 1].set_title('Computational Cost Comparison by Model and HPO Method', fontweight='bold', fontsize=12)
        axes[0, 1].set_ylabel('Search Time (seconds)', fontweight='bold')
        axes[0, 1].set_xlabel('Model Type', fontweight='bold')
        axes[0, 1].legend(title='HPO Method', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].tick_params(axis='x', rotation=0)
        
        # 3. Average Accuracy by HPO Method
        hpo_avg = self.results.groupby('hpo_method')['test_accuracy'].mean().sort_values(ascending=False)
        hpo_avg.plot(kind='barh', ax=axes[1, 0], color='skyblue', edgecolor='black', linewidth=1.2)
        axes[1, 0].set_title('Average Performance by HPO Method', fontweight='bold', fontsize=12)
        axes[1, 0].set_xlabel('Average Test Accuracy', fontweight='bold')
        axes[1, 0].set_ylabel('HPO Method', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        axes[1, 0].set_xlim(0.90, 1.00)  # 90%~100% 범위로 확대
        
        # Add value labels
        for i, v in enumerate(hpo_avg):
            axes[1, 0].text(v + 0.002, i, f'{v:.4f}', va='center', fontweight='bold')
        
        # 4. Average Accuracy by Model
        model_avg = self.results.groupby('model')['test_accuracy'].mean().sort_values(ascending=False)
        model_avg.plot(kind='barh', ax=axes[1, 1], color='lightcoral', edgecolor='black', linewidth=1.2)
        axes[1, 1].set_title('Average Performance by Model Type', fontweight='bold', fontsize=12)
        axes[1, 1].set_xlabel('Average Test Accuracy', fontweight='bold')
        axes[1, 1].set_ylabel('Model Type', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        axes[1, 1].set_xlim(0.90, 1.00)  # 90%~100% 범위로 확대
        
        # Add value labels
        for i, v in enumerate(model_avg):
            axes[1, 1].text(v + 0.002, i, f'{v:.4f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / 'model_hpo_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        return fig
    
    def plot_confusion_matrices(self, save=True):
        """Plot confusion matrices for best result of each model."""
        models = self.results['model'].unique()
        n_models = len(models)
        
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for i, model in enumerate(models):
            # Get best result for this model
            model_results = self.results[self.results['model'] == model]
            best_idx = int(model_results['test_accuracy'].idxmax())  # 명시적 int 변환
            best_result = self.results.loc[best_idx]
            
            # Get confusion matrix
            cm = np.array(best_result['confusion_matrix'])
            
            # Plot
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=[str(i) for i in range(10)],  # 문자열 리스트로 변환
                yticklabels=[str(i) for i in range(10)],  # 문자열 리스트로 변환
                ax=axes[i],
                cbar_kws={'label': 'Count'}
            )
            
            axes[i].set_title(
                f'{model.upper()}\n{best_result["hpo_method"].replace("_", " ").title()} (Accuracy: {best_result["test_accuracy"]:.4f})',
                fontweight='bold',
                fontsize=11
            )
            axes[i].set_ylabel('True Label', fontweight='bold', fontsize=10)
            axes[i].set_xlabel('Predicted Label', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / 'confusion_matrices.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        return fig
    
    def plot_optuna_histories(self, optuna_studies: Optional[Dict[str, Any]], save: bool = True):
        """
        Plot Optuna optimization histories.
        
        Args:
            optuna_studies: Dictionary of {model_name: study} for Optuna results
            save: Whether to save the plot
        """
        if not optuna_studies:
            print("No Optuna studies to plot")
            return None
        
        n_studies = len(optuna_studies)
        fig, axes = plt.subplots(1, n_studies, figsize=(7*n_studies, 5))
        if n_studies == 1:
            axes = [axes]
        
        for i, (model_name, study) in enumerate(optuna_studies.items()):
            trials = study.trials
            scores = [trial.value for trial in trials if trial.value is not None]
            
            if len(scores) == 0:
                continue
            
            # Plot trial scores
            axes[i].plot(scores, 'o-', alpha=0.6, label='Trial Score', markersize=4)
            
            # Plot best score line
            best_scores = [max(scores[:j+1]) for j in range(len(scores))]
            axes[i].plot(best_scores, 'r-', linewidth=2.5, label='Best Score')
            
            axes[i].set_title(f'{model_name.upper()}\nOptuna Optimization Progress', fontweight='bold', fontsize=12)
            axes[i].set_xlabel('Trial Number', fontweight='bold', fontsize=10)
            axes[i].set_ylabel('Validation Accuracy', fontweight='bold', fontsize=10)
            axes[i].legend(frameon=True, shadow=True)
            axes[i].grid(True, alpha=0.3)
            
            # Add final best score annotation
            axes[i].text(
                len(scores)-1, best_scores[-1],
                f' Best: {best_scores[-1]:.4f}',
                fontsize=10,
                fontweight='bold',
                va='bottom'
            )
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / 'optuna_histories.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        return fig
    
    def create_all_visualizations(self, optuna_studies=None):
        """Create all visualizations for publication."""
        print("\nGenerating publication-ready visualizations...")
        print("-" * 60)
        
        self.plot_accuracy_vs_time(save=True)
        self.plot_model_hpo_comparison(save=True)
        self.plot_confusion_matrices(save=True)
        
        if optuna_studies:
            self.plot_optuna_histories(optuna_studies, save=True)
        
        print("-" * 60)
        print(f"All visualizations saved to: {self.save_dir}\n")

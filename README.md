# MNIST Hyperparameter Optimization (HPO) Comparison

## 📋 Project Overview

An experimental project comparing various hyperparameter optimization techniques for MNIST handwritten digit classification.

### Experiment Configuration
- **Models**: CNN (PyTorch), Random Forest, XGBoost (3 in total)
- **HPO Methods**: Grid Search, Random Search, Bayesian Optimization, Optuna (4 in total)
- **Total Combinations**: 3 models × 4 HPO methods = **12 combinations**
- **Dataset**: MNIST (70,000 images, 10-class classification)

### Tuning Hyperparameters
- **Learning Rate**: Adjusts model learning speed
- **Epochs**: Number of training iterations
- **Model Architecture Parameters**: 
  - CNN: conv_layers, filters, dense_units, dropout
  - RandomForest: n_estimators, max_depth, min_samples_split
  - XGBoost: max_depth, subsample, colsample_bytree, regularization

---

## 🏗️ Project Structure

```text
HPO/
├── config/
│   └── hpo_config.yaml          # Experiment config file
├── data/
│   ├── __init__.py
│   └── data_loader.py           # MNIST data loader
├── models/
│   ├── __init__.py
│   ├── cnn_model.py             # CNN model builder
│   ├── random_forest_model.py   # RandomForest model builder
│   └── xgboost_model.py         # XGBoost model builder
├── hpo/
│   ├── __init__.py
│   ├── grid_search.py           # Grid Search implementation
│   ├── random_search.py         # Random Search implementation
│   ├── bayesian_optimization.py # Bayesian Optimization implementation
│   └── optuna_optimizer.py      # Optuna implementation
├── experiments/
│   ├── __init__.py
│   ├── experiment_tracker.py    # Experiment result tracker
│   └── experiment_runner.py     # Experiment execution pipeline
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # Evaluation metrics
│   └── visualizer.py            # Visualization
├── results/
│   ├── plots/                   # Generated charts
│   ├── logs/                    # Experiment logs
│   └── checkpoints/             # Saved models (12 in total)
├── requirements.txt             # Package dependencies
├── main.py                      # Main execution script
└── README.md                    # Project documentation (this file)
```

---

## 🚀 Installation

### 1. Python Environment Setup
Requires Python 3.8 or higher.

```bash
# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### 2. Required Packages
```text
numpy>=1.21.0
scikit-learn>=1.0.0
torch>=2.0.0
torchvision>=0.15.0
xgboost>=1.6.0
optuna>=3.0.0
scikit-optimize>=0.9.0
matplotlib>=3.5.0
seaborn>=0.12.0
plotly>=5.10.0
pyyaml>=6.0
pandas>=1.4.0
tqdm>=4.64.0
```

---

## 💻 Usage

### Run Full Experiment
```bash
python main.py
```

### Customize Configuration File
```bash
python main.py --config custom_config.yaml
```

### Regenerate Visualizations Only (Skip experiments)
```bash
python main.py --skip-experiments
```

---

## ⚙️ Experiment Configuration

You can adjust experiment parameters in the `config/hpo_config.yaml` file:

```yaml
experiment:
  name: "mnist_hpo_comparison"
  seed: 42                    # Random seed for reproducibility
  timeout: 18000              # Total experiment time limit (5 hours)
  save_all_checkpoints: true  # Save all 12 models

execution_order:
  # Execute faster experiments first
  - model: "random_forest"
    hpo_methods: ["random_search", "optuna", "bayesian_optimization", "grid_search"]
  - model: "xgboost"
    hpo_methods: ["random_search", "optuna", "bayesian_optimization", "grid_search"]
  - model: "cnn"
    hpo_methods: ["random_search", "optuna", "bayesian_optimization", "grid_search"]
```

---

## 📊 Experiment Results

After the experiment is completed, the following files will be generated in the `results/` directory:

### 1. Result Files
- `experiment_results.json`: Overall results (JSON format)
- `experiment_results.csv`: Overall results (CSV format)
- `summary_report.txt`: Summary report

### 2. Visualizations
`results/plots/` directory:
- `accuracy_vs_time.png`: Scatter plot of Accuracy vs Search Time
- `model_hpo_comparison.png`: Bar chart comparing performance by model and HPO
- `confusion_matrices.png`: Confusion matrices of the best performing models (3 in total)
- `optuna_histories.png`: Optuna optimization history (3 in total)

### 3. Model Checkpoints
`results/checkpoints/` directory:
- `random_forest_grid_search.pkl`
- `random_forest_random_search.pkl`
- `random_forest_bayesian_optimization.pkl`
- `random_forest_optuna.pkl`
- `xgboost_grid_search.pkl`
- ... (12 models in total)

### 4. Logs
`results/logs/` directory:
- `experiment_YYYYMMDD_HHMMSS.log`: Detailed execution logs

---

## 📈 Result Interpretation

### HPO Methods Comparison
1. **Grid Search**
   - **Pros**: Exhaustive search, high reproducibility
   - **Cons**: Time-consuming
   - **Best for**: Small parameter spaces

2. **Random Search**
   - **Pros**: Fast execution, reasonable performance
   - **Cons**: Optimal point not guaranteed
   - **Best for**: Establishing a quick baseline

3. **Bayesian Optimization**
   - **Pros**: Efficient search, good performance
   - **Cons**: Complex implementation
   - **Best for**: Medium-scale search spaces

4. **Optuna**
   - **Pros**: Highly efficient, fast due to pruning
   - **Cons**: Requires initial setup
   - **Best for**: Large-scale hyperparameter search

### Evaluation Metrics
- **Test Accuracy**: Accuracy on the test dataset
- **CV Score**: Cross-validation score
- **Search Time**: Time taken for HPO search
- **F1 Score**: Harmonic mean of Precision and Recall
- **Confusion Matrix**: Classification performance per class

---

## 🔧 Customization

### Adding a New Model
1. Create a new model file in the `models/` directory.
2. Add model configuration to `config/hpo_config.yaml`.
3. Add execution logic to `experiments/experiment_runner.py`.

### Modifying Hyperparameter Ranges
Edit the `hyperparameters` section in `config/hpo_config.yaml`:
```yaml
hyperparameters:
  cnn:
    optuna:
      learning_rate: [0.0001, 0.01]  # Adjust range
      epochs: [10, 50]               # Adjust range
```

### Changing the Dataset
Modify `data/data_loader.py` to load a different dataset.

---

## 📝 Reproducing Experiments

To achieve the exact same results:
1. Maintain `seed: 42` in `config/hpo_config.yaml`.
2. Use the same package versions (`requirements.txt`).
3. Maintain the same execution order.

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
- Decrease `batch_size` in `config/hpo_config.yaml`.
- Decrease the number of CNN epochs.
- Run experiments sequentially.

### Execution Timeout
- Reduce Grid Search parameter ranges.
- Decrease `n_trials`/`n_iter` for Optuna/Random Search.
- Increase overall `timeout`.

### Package Installation Errors
```bash
# PyTorch installation (depends on your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Scikit-optimize installation issue
pip install scikit-optimize --no-deps
pip install scipy scikit-learn
```

---

## 📚 References

### HPO Methods
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Scikit-Optimize](https://scikit-optimize.github.io/)
- [Hyperparameter Optimization Survey](https://arxiv.org/abs/1810.05934)

### Models
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TorchVision Documentation](https://pytorch.org/vision/stable/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## 📄 License

This project is freely available for educational and research purposes.

## 👥 Contribution

Bug reports, feature suggestions, and Pull Requests are welcome!

<br>

> **Start Experiment**: `python main.py`  
> **Estimated Time**: Approx. 3-5 hours (depending on hardware)  
> **Check Results**: `results/summary_report.txt`

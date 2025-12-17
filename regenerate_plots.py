"""그래프 재생성 스크립트"""
import json
import pandas as pd
from evaluation.visualizer import ResultVisualizer

# 결과 로드
with open('results/mnist_hpo_comparison_results.json', 'r') as f:
    results = json.load(f)

df = pd.DataFrame(results)

# 시각화 생성
visualizer = ResultVisualizer('results')
visualizer.results = df

print("Generating model_hpo_comparison.png with y-axis [90%, 100%]...")
visualizer.plot_model_hpo_comparison(save=True)
print("✅ Complete!")

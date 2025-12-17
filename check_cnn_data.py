import pandas as pd
import json

# JSON 파일 확인
with open('results/mnist_hpo_comparison_results.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
cnn = df[df['model']=='cnn'][['hpo_method', 'search_time', 'test_accuracy']].copy()
cnn['time_hours'] = cnn['search_time'] / 3600
cnn = cnn.sort_values('search_time')

print("="*80)
print("CNN 실제 데이터 (시간 순)")
print("="*80)
print(cnn.to_string(index=False))

print("\n" + "="*80)
print("전체 데이터 개수:")
print(f"  Total: {len(df)}")
print(f"  CNN: {len(cnn)}")
print(f"  RandomForest: {len(df[df['model']=='random_forest'])}")
print(f"  XGBoost: {len(df[df['model']=='xgboost'])}")
print("="*80)

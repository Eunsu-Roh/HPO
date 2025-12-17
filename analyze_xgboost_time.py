import json
import pandas as pd

# 결과 로드
with open('results/mnist_hpo_comparison_results.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# XGBoost만 필터링
xgb = df[df['model']=='xgboost'][['hpo_method', 'search_time', 'test_accuracy']].copy()
xgb['search_time_hours'] = xgb['search_time'] / 3600
xgb = xgb.sort_values('search_time', ascending=False)

print("="*80)
print("XGBoost 실행 시간 비교")
print("="*80)
for idx, row in xgb.iterrows():
    print(f"\n{row['hpo_method']:25s}:")
    print(f"  시간: {row['search_time']:10.1f}초 = {row['search_time']/60:8.1f}분 = {row['search_time_hours']:6.2f}시간")
    print(f"  정확도: {row['test_accuracy']:.4f} ({row['test_accuracy']*100:.2f}%)")

print("\n" + "="*80)
print(f"가장 오래 걸린 방법: {xgb.iloc[0]['hpo_method']}")
print(f"가장 빠른 방법: {xgb.iloc[-1]['hpo_method']}")
print(f"시간 차이: {xgb.iloc[0]['search_time_hours'] - xgb.iloc[-1]['search_time_hours']:.2f}시간 ({(xgb.iloc[0]['search_time'] - xgb.iloc[-1]['search_time'])/3600:.1f}h)")
print("="*80)

# MNIST Hyperparameter Optimization (HPO) Comparison

## 📋 프로젝트 개요

MNIST 손글씨 숫자 분류를 위한 하이퍼파라미터 최적화 기법 비교 실험 프로젝트입니다.

### 실험 구성
- **모델**: CNN (PyTorch), Random Forest, XGBoost (총 3개)
- **HPO 기법**: Grid Search, Random Search, Bayesian Optimization, Optuna (총 4개)
- **총 실험 조합**: 3 models × 4 HPO methods = **12개 조합**
- **데이터셋**: MNIST (70,000개 이미지, 10-class 분류)

### 튜닝 하이퍼파라미터
- **Learning Rate**: 모델 학습 속도 조정
- **Epochs**: 학습 반복 횟수
- **Model Architecture Parameters**: 
  - CNN: conv_layers, filters, dense_units, dropout
  - RandomForest: n_estimators, max_depth, min_samples_split
  - XGBoost: max_depth, subsample, colsample_bytree, regularization

## 🏗️ 프로젝트 구조

```
HPO/
├── config/
│   └── hpo_config.yaml          # 실험 설정 파일
├── data/
│   ├── __init__.py
│   └── data_loader.py           # MNIST 데이터 로더
├── models/
│   ├── __init__.py
│   ├── cnn_model.py             # CNN 모델 빌더
│   ├── random_forest_model.py   # RandomForest 모델 빌더
│   └── xgboost_model.py         # XGBoost 모델 빌더
├── hpo/
│   ├── __init__.py
│   ├── grid_search.py           # Grid Search 구현
│   ├── random_search.py         # Random Search 구현
│   ├── bayesian_optimization.py # Bayesian Optimization 구현
│   └── optuna_optimizer.py      # Optuna 구현
├── experiments/
│   ├── __init__.py
│   ├── experiment_tracker.py    # 실험 결과 추적
│   └── experiment_runner.py     # 실험 실행 파이프라인
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # 평가 메트릭
│   └── visualizer.py            # 시각화
├── results/
│   ├── plots/                   # 생성된 차트
│   ├── logs/                    # 실험 로그
│   └── checkpoints/             # 저장된 모델 (12개)
├── requirements.txt             # 패키지 의존성
├── main.py                      # 메인 실행 스크립트
└── README.md                    # 프로젝트 문서 (이 파일)
```

## 🚀 설치 방법

### 1. Python 환경 설정
Python 3.8 이상 필요

```powershell
# 가상환경 생성 (선택사항)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 패키지 설치
pip install -r requirements.txt
```

### 2. 필수 패키지
```
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

## 💻 실행 방법

### 전체 실험 실행
```powershell
python main.py
```

### 설정 파일 커스터마이징
```powershell
python main.py --config custom_config.yaml
```

### 시각화만 재생성 (실험은 스킵)
```powershell
python main.py --skip-experiments
```

## ⚙️ 실험 설정

`config/hpo_config.yaml` 파일에서 실험 파라미터 조정 가능:

```yaml
experiment:
  name: "mnist_hpo_comparison"
  seed: 42                    # 재현성을 위한 랜덤 시드
  timeout: 18000              # 전체 실험 시간 제한 (5시간)
  save_all_checkpoints: true  # 12개 모델 모두 저장

execution_order:
  # 빠른 실험부터 실행
  - model: "random_forest"
    hpo_methods: ["random_search", "optuna", "bayesian_optimization", "grid_search"]
  - model: "xgboost"
    hpo_methods: ["random_search", "optuna", "bayesian_optimization", "grid_search"]
  - model: "cnn"
    hpo_methods: ["random_search", "optuna", "bayesian_optimization", "grid_search"]
```

## 📊 실험 결과

실험 완료 후 `results/` 디렉토리에 다음 파일들이 생성됩니다:

### 1. 결과 파일
- `experiment_results.json`: 전체 결과 (JSON 형식)
- `experiment_results.csv`: 전체 결과 (CSV 형식)
- `summary_report.txt`: 요약 리포트

### 2. 시각화
`results/plots/` 디렉토리:
- `accuracy_vs_time.png`: Accuracy vs Search Time 산점도
- `model_hpo_comparison.png`: 모델별/HPO별 성능 비교 막대 차트
- `confusion_matrices.png`: 각 모델의 최고 성능 혼동 행렬 (3개)
- `optuna_histories.png`: Optuna 최적화 과정 (3개)

### 3. 모델 체크포인트
`results/checkpoints/` 디렉토리:
- `random_forest_grid_search.pkl`
- `random_forest_random_search.pkl`
- `random_forest_bayesian_optimization.pkl`
- `random_forest_optuna.pkl`
- `xgboost_grid_search.pkl`
- ... (총 12개 모델)

### 4. 로그
`results/logs/` 디렉토리:
- `experiment_YYYYMMDD_HHMMSS.log`: 상세 실행 로그

## 📈 결과 해석

### HPO 기법 비교
1. **Grid Search**
   - 장점: 완전 탐색, 재현성 높음
   - 단점: 시간이 오래 걸림
   - 적합한 경우: 파라미터 공간이 작을 때

2. **Random Search**
   - 장점: 빠른 실행, 합리적인 성능
   - 단점: 최적점 보장 안됨
   - 적합한 경우: 빠른 baseline 확보

3. **Bayesian Optimization**
   - 장점: 효율적인 탐색, 좋은 성능
   - 단점: 구현이 복잡함
   - 적합한 경우: 중간 규모 탐색

4. **Optuna**
   - 장점: 매우 효율적, pruning으로 빠름
   - 단점: 초기 설정 필요
   - 적합한 경우: 대규모 하이퍼파라미터 탐색

### 평가 메트릭
- **Test Accuracy**: 테스트 데이터 정확도
- **CV Score**: 교차 검증 점수
- **Search Time**: HPO 탐색 시간
- **F1 Score**: Precision과 Recall의 조화 평균
- **Confusion Matrix**: 클래스별 분류 성능

## 🔧 커스터마이징

### 새로운 모델 추가
1. `models/` 에 새 모델 파일 생성
2. `config/hpo_config.yaml`에 모델 설정 추가
3. `experiments/experiment_runner.py`에 실행 로직 추가

### 하이퍼파라미터 범위 변경
`config/hpo_config.yaml`의 `hyperparameters` 섹션 수정:
```yaml
hyperparameters:
  cnn:
    optuna:
      learning_rate: [0.0001, 0.01]  # 범위 조정
      epochs: [10, 50]               # 범위 조정
```

### 데이터셋 변경
`data/data_loader.py` 수정하여 다른 데이터셋 로드

## 📝 실험 재현

동일한 결과를 얻으려면:
1. `config/hpo_config.yaml`에서 `seed: 42` 유지
2. 동일한 패키지 버전 사용 (`requirements.txt`)
3. 동일한 실행 순서 유지

## 🐛 문제 해결

### 메모리 부족
- `config/hpo_config.yaml`에서 `batch_size` 감소
- CNN epochs 수 감소
- 실험을 순차적으로 실행

### 실행 시간 초과
- Grid Search 파라미터 범위 축소
- Optuna/Random Search `n_trials`/`n_iter` 감소
- 전체 `timeout` 증가

### 패키지 설치 오류
```powershell
# PyTorch 설치 (CUDA 버전에 따라 다름)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# scikit-optimize 설치 문제
pip install scikit-optimize --no-deps
pip install scipy scikit-learn
```

## 📚 참고 자료

### HPO 기법
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Scikit-Optimize](https://scikit-optimize.github.io/)
- [Hyperparameter Optimization Survey](https://arxiv.org/abs/1810.05934)

### 모델
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TorchVision Documentation](https://pytorch.org/vision/stable/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 자유롭게 사용 가능합니다.

## 👥 기여

버그 리포트, 기능 제안, Pull Request 환영합니다!

---

**실험 시작**: `python main.py`

**예상 소요 시간**: 약 3-5시간 (하드웨어에 따라 다름)

**결과 확인**: `results/summary_report.txt`

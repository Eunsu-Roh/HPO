# Hardware Optimization Settings

## 시스템 사양
- **CPU**: AMD Ryzen 9 5900X (12코어 / 24쓰레드)
- **RAM**: 64GB
- **GPU**: RTX 3080
- **OS**: Windows

## CPU 과부하 방지 설정

### 1. 전역 스레드 제한 (main.py)
```python
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
torch.set_num_threads(4)
```

### 2. HPO 메서드별 n_jobs 설정

#### Grid Search (hpo/grid_search.py)
- `n_jobs = 4` (12코어의 33%)
- Config: `n_jobs: 4`

#### Random Search (hpo/random_search.py)
- `n_jobs = 4` (12코어의 33%)
- Config: `n_jobs: 4`

#### Bayesian Optimization (hpo/bayesian_optimization.py)
- 모델 학습: `n_jobs = 6` (12코어의 50%)
- Cross-validation: `n_jobs = 4` (12코어의 33%)
- Config: `n_jobs: 1` (순차 실행)

#### Optuna (hpo/optuna_optimizer.py)
- RandomForest: `n_jobs = 6`, CV: `n_jobs = 2`
- XGBoost: `n_jobs = 6`, CV: `n_jobs = 2`
- Config: `n_jobs: 1` (순차 실행)

### 3. PyTorch DataLoader 설정
```python
DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,      # Windows 안정성
    pin_memory=False    # CPU 메모리 부담 감소
)
```

### 4. GPU 사용 (CNN 모델)
- CUDA 자동 감지 및 사용
- CPU fallback 지원
- 배치 크기: 32~256 (메모리에 맞게 자동 조정)

## 예상 리소스 사용량

### CPU 사용률
- **Grid/Random Search**: ~33% (4코어)
- **Bayesian Optimization**: ~50% (모델 6코어 + CV 4코어, 순차 실행)
- **Optuna**: ~67% (모델 6코어 + CV 2코어)
- **최대 동시 사용**: 8코어 (67%)

### 메모리
- MNIST 데이터셋: ~500MB
- RandomForest 모델: ~2GB
- XGBoost 모델: ~1GB
- CNN 모델 (GPU): ~500MB VRAM
- 총 예상: ~10GB RAM (64GB 중 16%)

### 실행 시간 예상 (12개 실험)
1. RandomForest (4개): ~1-2시간
2. XGBoost (4개): ~2-3시간
3. CNN (4개): ~1-2시간 (GPU 가속)
- **총 예상**: 4-7시간

## 안정성 확인 체크리스트

✅ **CPU 스레드 제한 설정 완료**
- main.py에서 환경변수 설정
- PyTorch 스레드 제한
- 모든 HPO 메서드 n_jobs 제한

✅ **메모리 관리**
- DataLoader num_workers=0 (Windows 안정성)
- pin_memory=False (CPU 메모리 절약)
- 배치 단위 처리로 메모리 효율화

✅ **중단 후 재개 기능**
- ExperimentTracker 자동 저장/로드
- 완료된 실험 자동 스킵
- JSON/CSV 백업

✅ **에러 처리**
- 개별 실험 실패 시 다음 실험 계속
- 전체 스택 트레이스 로깅
- 체크포인트 자동 저장

## 실행 방법

```powershell
# 환경 활성화
conda activate env

# 학습 시작 (중단된 지점부터 자동 재개)
python main.py

# 시각화만 생성 (실험 스킵)
python main.py --skip-experiments
```

## 모니터링 권장사항

1. **작업 관리자 모니터링**
   - CPU 사용률: 50-70% 목표
   - RAM 사용량: 20GB 미만
   - GPU 사용률 (CNN 학습 시)

2. **로그 확인**
   - `results/logs/` 디렉토리
   - 실시간 콘솔 출력

3. **중간 결과 확인**
   - `results/mnist_hpo_comparison_results.json`
   - `results/mnist_hpo_comparison_results.csv`

## 문제 발생 시

### 컴퓨터 여전히 느려질 경우
1. config/hpo_config.yaml에서 n_jobs를 더 낮춤 (4 → 2)
2. n_iter, n_trials 줄이기 (100 → 50)

### 메모리 부족
1. 배치 크기 줄이기 (config의 batch_size 범위)
2. CNN epochs 줄이기

### GPU 메모리 부족
1. 배치 크기 강제 축소
2. CPU로 폴백 (자동)

---
**마지막 업데이트**: 2025년 11월 27일
**안정성 검증**: Ryzen 9 5900X 12코어 시스템에서 최적화 완료

"""
간소화된 코드 사용 가이드
===========================

## 🎯 핵심 변경사항

1. **main.py의 run 함수 - 5개 인자만!**
   - is_preprocess: 전처리 여부
   - is_feature_engineering: 피처 엔지니어링 여부
   - cv_strategy: CV 전략
   - tuning_strategy: 튜닝 전략
   - ensemble_strategy: 앙상블 전략
   - is_save: 저장 여부

2. **통일된 CV 전략**
   - main.py에서 정한 cv_strategy가 앙상블 함수까지 일관되게 적용

3. **간결한 코드**
   - 불필요한 파라미터 제거
   - 핵심 기능만 유지

## 📝 사용 예시

### 1. 기본 사용 (빠른 프로토타입)

```python
from src.preprocessing import load_data
from main import run

df = load_data()

results = run(
    df=df,
    is_preprocess=True,
    is_feature_engineering=True,
    cv_strategy='stratified_kfold',  # 'stratified_kfold', 'kfold', None
    tuning_strategy=None,  # 튜닝 안 함 (빠름)
    ensemble_strategy='stacking',  # 'stacking', 'voting', 'logistic'
    is_save=True
)

print(f"평균 F1: {results['summary']['f1']['mean']:.4f}")
```

### 2. 튜닝 모드 (성능 최적화)

```python
results = run(
    df=df,
    is_preprocess=True,
    is_feature_engineering=True,
    cv_strategy='stratified_kfold',
    tuning_strategy='optuna',  # 👈 튜닝 활성화!
    ensemble_strategy='stacking',
    is_save=True
)
```

### 3. CV 전략 선택

```python
# Stratified KFold (불균형 데이터 추천)
results = run(df, cv_strategy='stratified_kfold', ...)

# 일반 KFold
results = run(df, cv_strategy='kfold', ...)

# CV 없이 단순 분할
results = run(df, cv_strategy=None, ...)
```

### 4. 앙상블 전략 선택

```python
# Stacking (추천, 성능 좋음)
results = run(df, ensemble_strategy='stacking', ...)

# Voting (빠름, 괜찮은 성능)
results = run(df, ensemble_strategy='voting', ...)

# Logistic Regression (베이스라인)
results = run(df, ensemble_strategy='logistic', ...)
```

## 🎮 실전 워크플로우

```python
# 1단계: 빠른 검증 (5분)
run(df, cv_strategy='stratified_kfold', tuning_strategy=None, 
    ensemble_strategy='voting')

# 2단계: Stacking 테스트 (10분)
run(df, cv_strategy='stratified_kfold', tuning_strategy=None,
    ensemble_strategy='stacking')

# 3단계: 최종 튜닝 (1-2시간)
run(df, cv_strategy='stratified_kfold', tuning_strategy='optuna',
    ensemble_strategy='stacking')
```

## 📊 변경 전/후 비교

### Before (복잡)
```python
run(
    df=df,
    is_preprocess=True,
    is_feature_engineering=True,
    model_type='stacking',
    cv_strategy='stratified_kfold',
    n_splits=5,
    use_custom_cv=False,
    do_tuning=True,
    tuning_strategy='optuna',
    tuning_before_cv=True,
    n_trials=50,
    is_save=True,
    save_dir='results'
)  # 😵 너무 많은 인자!
```

### After (간결)
```python
run(
    df=df,
    is_preprocess=True,
    is_feature_engineering=True,
    cv_strategy='stratified_kfold',
    tuning_strategy='optuna',  # None이면 튜닝 안 함
    ensemble_strategy='stacking',
    is_save=True
)  # ✨ 깔끔!
```

## 💡 핵심 개념

### CV 전략 통일
- main.py에서 `cv_strategy='stratified_kfold'` 선택
- ↓
- 외부 CV (main.py): StratifiedKFold로 폴드 split
- ↓
- 내부 CV (ensemble.py): StackingClassifier도 같은 StratifiedKFold 사용
- ↓
- 튜닝 CV (tuner.py): 튜닝할 때도 같은 StratifiedKFold 사용

→ **일관된 CV 전략!**

### tuning_strategy = None의 의미
- None: 기본 하이퍼파라미터 사용 (빠름)
- 'optuna', 'grid_search' 등: 튜닝 수행 (느리지만 성능 좋음)

→ **별도의 do_tuning 불필요!**
"""

print(__doc__)

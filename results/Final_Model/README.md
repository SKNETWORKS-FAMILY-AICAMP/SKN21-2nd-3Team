# Final Model

이 디렉토리는 학습된 최종 모델들을 저장합니다.

## 디렉토리 구조

```
Final_Model/
├── best/                          # 최종 선택된 모델
│   └── final_model.joblib
├── ensemble/                      # 앙상블 전략별 모델
│   ├── stacking_model.joblib
│   └── voting_model.joblib
├── experiments/                   # 실험 모델들
│   └── feature_selection/         # Feature selection 실험 결과
│       ├── Exclude_Features_selection_00_stacking_model.joblib
│       ├── Exclude_Features_selection_01_stacking_model.joblib
│       ├── Exclude_Features_selection_10_stacking_model.joblib
│       └── Exclude_Features_selection_11_stacking_model.joblib
└── README.md                      # 이 파일
```

## 파일 설명

### 최종 모델 (`best/`)

- **`final_model.joblib`**: 최종적으로 선택된 최고 성능 모델
  - MetaScore 기준으로 선정된 최적 모델
  - 프로덕션 환경에서 사용하는 모델

### 앙상블 모델 (`ensemble/`)

- **`stacking_model.joblib`**: Stacking 앙상블 모델
  - Base Models: Random Forest, XGBoost, LightGBM
  - Meta-Learner: Logistic Regression
  - Streamlit 대시보드에서 사용 중 (`results/streamlit/utils.py`)

- **`voting_model.joblib`**: Voting 앙상블 모델
  - Base Models: Random Forest, XGBoost, LightGBM
  - Hard Voting 방식

### 실험 모델 (`experiments/feature_selection/`)

Feature selection과 feature engineering 조합별 실험 결과:

- **`Exclude_Features_selection_00_stacking_model.joblib`**
  - Feature selection: ❌ (0)
  - Feature engineering: ❌ (0)

- **`Exclude_Features_selection_01_stacking_model.joblib`**
  - Feature selection: ❌ (0)
  - Feature engineering: ✅ (1)

- **`Exclude_Features_selection_10_stacking_model.joblib`**
  - Feature selection: ✅ (1)
  - Feature engineering: ❌ (0)

- **`Exclude_Features_selection_11_stacking_model.joblib`**
  - Feature selection: ✅ (1)
  - Feature engineering: ✅ (1)

## 모델 사용 방법

### Python에서 모델 로드

```python
import joblib

# 최종 모델 로드
final_model = joblib.load('results/Final_Model/best/final_model.joblib')

# Stacking 모델 로드
stacking_model = joblib.load('results/Final_Model/ensemble/stacking_model.joblib')

# 예측 수행
predictions = final_model.predict(X_test)
probabilities = final_model.predict_proba(X_test)
```

### Streamlit에서 모델 사용

Streamlit 대시보드에서는 `results/streamlit/utils.py`의 `load_model()` 함수를 통해 자동으로 모델을 로드합니다:

```python
from results.streamlit.utils import load_model

model = load_model()  # ensemble/stacking_model.joblib 자동 로드
```

## 모델 학습 정보

자세한 모델 학습 과정 및 평가 지표는 `results/Modeling/README.md`를 참고하세요.

### 주요 특징

- **앙상블 전략**: Stacking Ensemble
- **Base Models**: Random Forest, XGBoost, LightGBM
- **Meta-Learner**: Logistic Regression
- **CV 전략**: Stratified K-Fold (K=5)
- **튜닝 방법**: Optuna (베이지안 최적화)
- **평가 지표**: MetaScore = 0.5×F2 + 0.3×PR-AUC + 0.2×ROC-AUC

## 모델 저장 위치 변경 이력

이전에는 모든 모델이 `results/Final_Model/` 루트에 저장되어 있었으나, GitHub 관례에 맞게 다음과 같이 재구성되었습니다:

- 최종 모델 → `best/`
- 앙상블 모델 → `ensemble/`
- 실험 모델 → `experiments/feature_selection/`

이 구조를 통해 모델의 용도와 중요도를 명확히 구분할 수 있습니다.


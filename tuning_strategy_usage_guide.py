"""
📚 튜닝 전략 사용법 가이드

이 파일은 ensemble.py의 tuning_strategy 파라미터를 
main.py에서 어떻게 사용하는지 보여주는 예시입니다.
"""

from src.ensemble import train_voting_ensemble, train_stacking_ensemble, evaluate_model
import pandas as pd
from sklearn.model_selection import train_test_split

# =============================================================================
# 📊 데이터 준비 (예시)
# =============================================================================
# 실제 코드에서는 여러분의 데이터 로딩 코드를 사용하세요
# X_train, X_test, y_train, y_test = ...

# =============================================================================
# 🎯 사용법 1: 기본 모드 (튜닝 없음 - 빠름)
# =============================================================================
print("=" * 80)
print("방법 1: 기본 파라미터 사용 (튜닝 없음) - 개발/프로토타입에 적합")
print("=" * 80)

# 튜닝 없이 기본 파라미터로 빠르게 실험
model_basic = train_voting_ensemble(
    X_train=X_train,
    y_train=y_train,
    rf_weight=1,
    xgb_weight=2,
    lgbm_weight=2,
    voting='soft'
    # tuning_strategy=None (기본값) - 튜닝 안 함
)

# 평가
metrics_basic = evaluate_model(model_basic, X_test, y_test)
print(f"\n기본 모드 F1 Score: {metrics_basic['f1']:.4f}")


# =============================================================================
# 🚀 사용법 2: Optuna 튜닝 (추천!)
# =============================================================================
print("\n" + "=" * 80)
print("방법 2: Optuna 하이퍼파라미터 튜닝 - 최종 성능 최적화에 적합")
print("=" * 80)

# Optuna를 사용해 자동으로 최적 하이퍼파라미터 찾기
model_tuned = train_voting_ensemble(
    X_train=X_train,
    y_train=y_train,
    rf_weight=1,
    xgb_weight=2,
    lgbm_weight=2,
    voting='soft',
    tuning_strategy='optuna',  # 👈 핵심!
    cv=5,                      # 교차검증 폴드 수
    n_trials=50                # 튜닝 시도 횟수 (많을수록 좋지만 시간 오래 걸림)
)

metrics_tuned = evaluate_model(model_tuned, X_test, y_test)
print(f"\nOptuna 튜닝 모드 F1 Score: {metrics_tuned['f1']:.4f}")
print(f"성능 향상: {(metrics_tuned['f1'] - metrics_basic['f1']):.4f}")


# =============================================================================
# 📚 사용법 3: Stacking Ensemble + 튜닝
# =============================================================================
print("\n" + "=" * 80)
print("방법 3: Stacking Ensemble + Optuna 튜닝")
print("=" * 80)

model_stacking = train_stacking_ensemble(
    X_train=X_train,
    y_train=y_train,
    cv_folds=5,                # Stacking 내부 CV
    tuning_strategy='optuna',  # 베이스 모델 튜닝
    cv_tuning=3,               # 튜닝 시 CV (시간 절약을 위해 줄임)
    n_trials=30                # 튜닝 시도 횟수
)

metrics_stacking = evaluate_model(model_stacking, X_test, y_test)
print(f"\nStacking + 튜닝 F1 Score: {metrics_stacking['f1']:.4f}")


# =============================================================================
# 💡 main.py에서 tuning_strategy를 인자로 받는 방법
# =============================================================================
def train_ensemble_model(
    X_train, 
    y_train, 
    ensemble_type='voting',        # 'voting' or 'stacking'
    tuning_strategy=None,          # None, 'optuna', 'grid_search', 'random_search'
    n_trials=50
):
    """
    앙상블 모델을 학습하는 통합 함수.
    
    💡 설계 포인트:
    - ensemble_type과 tuning_strategy를 인자로 받아서
    - 다양한 조합을 쉽게 실험할 수 있게 함
    
    Args:
        X_train: 훈련 데이터
        y_train: 타겟 데이터
        ensemble_type: 'voting' 또는 'stacking'
        tuning_strategy: 튜닝 방법 (None이면 기본 파라미터)
        n_trials: 튜닝 시도 횟수
    
    Returns:
        학습된 모델
        
    예시:
        # CLI에서 사용
        >>> model = train_ensemble_model(
        ...     X_train, y_train, 
        ...     ensemble_type='voting',
        ...     tuning_strategy='optuna'
        ... )
    """
    
    if ensemble_type == 'voting':
        model = train_voting_ensemble(
            X_train=X_train,
            y_train=y_train,
            tuning_strategy=tuning_strategy,
            n_trials=n_trials
        )
    elif ensemble_type == 'stacking':
        model = train_stacking_ensemble(
            X_train=X_train,
            y_train=y_train,
            tuning_strategy=tuning_strategy,
            n_trials=n_trials
        )
    else:
        raise ValueError(f"Unknown ensemble_type: {ensemble_type}")
    
    return model


# =============================================================================
# 🎮 실전 사용 예시: Config 파일로 관리
# =============================================================================
"""
config.py 파일에서:

ENSEMBLE_CONFIG = {
    'ensemble_type': 'voting',     # 또는 'stacking'
    'tuning_strategy': 'optuna',   # 또는 None, 'grid_search', 'random_search'
    'n_trials': 50,                # 튜닝 시도 횟수
    'cv': 5                        # 교차검증 폴드
}

main.py에서:

from config import ENSEMBLE_CONFIG

model = train_ensemble_model(
    X_train, y_train, 
    **ENSEMBLE_CONFIG  # config를 그대로 전달
)
"""


# =============================================================================
# 📝 성능 비교 요약
# =============================================================================
print("\n" + "=" * 80)
print("📊 성능 비교 요약")
print("=" * 80)
print(f"기본 모드 (튜닝 없음):      F1={metrics_basic['f1']:.4f}, Recall={metrics_basic['recall']:.4f}")
print(f"Voting + Optuna 튜닝:       F1={metrics_tuned['f1']:.4f}, Recall={metrics_tuned['recall']:.4f}")
print(f"Stacking + Optuna 튜닝:     F1={metrics_stacking['f1']:.4f}, Recall={metrics_stacking['recall']:.4f}")
print("=" * 80)


# =============================================================================
# 🎯 추천 워크플로우
# =============================================================================
"""
💡 실전 프로젝트 워크플로우 추천:

1단계: 빠른 프로토타입 (tuning_strategy=None)
   - 데이터 전처리, 피처 엔지니어링 검증
   - 여러 앙상블 방법 빠르게 비교
   - 시간: 몇 분

2단계: 중간 튜닝 (tuning_strategy='random_search', n_trials=30)
   - 괜찮은 하이퍼파라미터 찾기
   - 시간: 10-30분

3단계: 최종 튜닝 (tuning_strategy='optuna', n_trials=100)  
   - 제출 전 최종 성능 최적화
   - 시간: 1-2시간

💡 시간/성능 트레이드오프:
- 개발 단계: tuning_strategy=None
- 검증 단계: tuning_strategy='random_search', n_trials=30
- 최종 제출: tuning_strategy='optuna', n_trials=100
"""

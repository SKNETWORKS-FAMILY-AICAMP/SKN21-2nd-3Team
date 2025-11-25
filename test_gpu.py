"""
XGBoost/LightGBM GPU 가속 테스트 스크립트
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

def test_xgboost_gpu():
    """XGBoost GPU vs CPU 속도 비교"""
    print("\n" + "="*60)
    print("🚀 XGBoost GPU 테스트")
    print("="*60)
    
    # 큰 데이터셋 생성
    X, y = make_classification(
        n_samples=100000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"데이터 크기: {X_train.shape}")
    
    # CPU 모드
    print("\n⏱️  CPU 모드 학습 중...")
    start = time.time()
    model_cpu = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        tree_method='hist',
        device='cpu',
        random_state=42
    )
    model_cpu.fit(X_train, y_train)
    cpu_time = time.time() - start
    cpu_score = model_cpu.score(X_test, y_test)
    print(f"   ✅ CPU 시간: {cpu_time:.2f}초, 정확도: {cpu_score:.4f}")
    
    # GPU 모드 (CUDA 사용 가능한 경우)
    print("\n⚡ GPU 모드 학습 중...")
    try:
        start = time.time()
        model_gpu = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            tree_method='hist',
            device='cuda',
            random_state=42
        )
        model_gpu.fit(X_train, y_train)
        gpu_time = time.time() - start
        gpu_score = model_gpu.score(X_test, y_test)
        print(f"   ✅ GPU 시간: {gpu_time:.2f}초, 정확도: {gpu_score:.4f}")
        print(f"\n🎉 속도 향상: {cpu_time/gpu_time:.2f}x 빠름!")
    except Exception as e:
        print(f"   ❌ GPU 실행 실패: {str(e)}")
        print("   💡 CUDA가 설치되어 있지 않거나 GPU를 지원하지 않는 XGBoost입니다.")

def test_lightgbm_gpu():
    """LightGBM GPU vs CPU 속도 비교"""
    print("\n" + "="*60)
    print("💡 LightGBM GPU 테스트")
    print("="*60)
    
    # 데이터셋 생성
    X, y = make_classification(
        n_samples=100000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"데이터 크기: {X_train.shape}")
    
    # CPU 모드
    print("\n⏱️  CPU 모드 학습 중...")
    start = time.time()
    model_cpu = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        device='cpu',
        random_state=42,
        verbosity=-1
    )
    model_cpu.fit(X_train, y_train)
    cpu_time = time.time() - start
    cpu_score = model_cpu.score(X_test, y_test)
    print(f"   ✅ CPU 시간: {cpu_time:.2f}초, 정확도: {cpu_score:.4f}")
    
    # GPU 모드
    print("\n⚡ GPU 모드 학습 중...")
    try:
        start = time.time()
        model_gpu = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            device='gpu',
            random_state=42,
            verbosity=-1
        )
        model_gpu.fit(X_train, y_train)
        gpu_time = time.time() - start
        gpu_score = model_gpu.score(X_test, y_test)
        print(f"   ✅ GPU 시간: {gpu_time:.2f}초, 정확도: {gpu_score:.4f}")
        print(f"\n🎉 속도 향상: {cpu_time/gpu_time:.2f}x 빠름!")
    except Exception as e:
        print(f"   ❌ GPU 실행 실패: {str(e)}")
        print("   💡 GPU 버전 LightGBM이 필요합니다.")
        print("   설치: pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 GPU 가속 테스트")
    print("="*60)
    
    test_xgboost_gpu()
    test_lightgbm_gpu()
    
    print("\n" + "="*60)
    print("✅ 테스트 완료!")
    print("="*60 + "\n")


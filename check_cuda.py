"""
GPU 및 CUDA 설정 확인 스크립트
이 스크립트는 시스템의 CUDA 및 GPU 설정을 종합적으로 진단합니다.
"""

import sys
import subprocess
import platform
import os
import glob

def print_header(title):
    """섹션 헤더 출력"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_success(message):
    """성공 메시지 출력"""
    print(f"[OK] {message}")

def print_error(message):
    """에러 메시지 출력"""
    print(f"[ERR] {message}")

def print_info(message):
    """정보 메시지 출력"""
    print(f"[INFO] {message}")

def find_nvcc_windows():
    """Windows에서 nvcc 경로 찾기"""
    import os
    import glob
    
    # 일반적인 CUDA 설치 경로들
    cuda_base_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA",
    ]
    
    nvcc_paths = []
    for base_path in cuda_base_paths:
        if os.path.exists(base_path):
            # v*.* 패턴으로 버전 폴더 찾기
            version_dirs = glob.glob(os.path.join(base_path, "v*"))
            for version_dir in version_dirs:
                nvcc_path = os.path.join(version_dir, "bin", "nvcc.exe")
                if os.path.exists(nvcc_path):
                    nvcc_paths.append(nvcc_path)
    
    return nvcc_paths

def check_cuda_toolkit():
    """CUDA Toolkit 설치 확인"""
    print_header("1. CUDA Toolkit 확인")
    
    nvcc_cmd = 'nvcc'
    
    # Windows인 경우 직접 경로 찾기 시도
    if platform.system() == 'Windows':
        try:
            # 먼저 PATH에서 nvcc 찾기 시도
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                print_success("CUDA Toolkit이 PATH에서 발견되었습니다.")
                print(result.stdout)
                return True
        except FileNotFoundError:
            print_info("PATH에서 nvcc를 찾을 수 없습니다. 일반 설치 경로를 검색합니다...")
            
            # 일반적인 설치 경로에서 찾기
            nvcc_paths = find_nvcc_windows()
            if nvcc_paths:
                print_info(f"발견된 CUDA 설치: {len(nvcc_paths)}개")
                for nvcc_path in nvcc_paths:
                    try:
                        result = subprocess.run([nvcc_path, '--version'], 
                                              capture_output=True, 
                                              text=True, 
                                              timeout=5)
                        if result.returncode == 0:
                            print_success(f"CUDA Toolkit 발견: {nvcc_path}")
                            print(result.stdout)
                            print_info("\n[해결방법] PATH 환경변수에 다음 경로를 추가하세요:")
                            cuda_bin_path = os.path.dirname(nvcc_path)
                            print_info(f"  {cuda_bin_path}")
                            print_info("\n시스템 환경변수 설정 후 터미널을 재시작해야 적용됩니다.")
                            return True
                    except Exception as e:
                        continue
                
                print_error("CUDA가 발견되었지만 실행할 수 없습니다.")
                return False
            else:
                print_error("CUDA Toolkit을 찾을 수 없습니다.")
                return False
    else:
        # Linux/Mac인 경우 기존 로직
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                print_success("CUDA Toolkit이 설치되어 있습니다.")
                print(result.stdout)
                return True
            else:
                print_error("CUDA Toolkit을 찾을 수 없습니다.")
                return False
        except FileNotFoundError:
            print_error("nvcc 명령어를 찾을 수 없습니다. CUDA Toolkit이 설치되지 않았거나 PATH에 없습니다.")
            return False
        except Exception as e:
            print_error(f"CUDA Toolkit 확인 중 오류: {str(e)}")
            return False

def find_nvidia_smi_windows():
    """Windows에서 nvidia-smi 경로 찾기"""
    nvidia_smi_paths = [
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]
    
    for path in nvidia_smi_paths:
        if os.path.exists(path):
            return path
    return None

def check_nvidia_smi():
    """NVIDIA GPU 드라이버 확인"""
    print_header("2. NVIDIA GPU 드라이버 확인")
    
    nvidia_smi_cmd = 'nvidia-smi'
    
    # Windows인 경우 직접 경로 찾기 시도
    if platform.system() == 'Windows':
        try:
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                print_success("NVIDIA 드라이버가 정상적으로 작동하고 있습니다.")
                print(result.stdout)
                return True
        except FileNotFoundError:
            print_info("PATH에서 nvidia-smi를 찾을 수 없습니다. 일반 설치 경로를 검색합니다...")
            
            nvidia_smi_path = find_nvidia_smi_windows()
            if nvidia_smi_path:
                try:
                    result = subprocess.run([nvidia_smi_path], 
                                          capture_output=True, 
                                          text=True, 
                                          timeout=5)
                    if result.returncode == 0:
                        print_success(f"NVIDIA 드라이버 발견: {nvidia_smi_path}")
                        print(result.stdout)
                        return True
                except Exception as e:
                    print_error(f"nvidia-smi 실행 실패: {str(e)}")
                    return False
            
            print_error("nvidia-smi를 찾을 수 없습니다. NVIDIA 드라이버가 설치되지 않았습니다.")
            return False
    else:
        # Linux/Mac인 경우 기존 로직
        try:
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                print_success("NVIDIA 드라이버가 정상적으로 작동하고 있습니다.")
                print(result.stdout)
                return True
            else:
                print_error("nvidia-smi 실행 실패")
                return False
        except FileNotFoundError:
            print_error("nvidia-smi를 찾을 수 없습니다. NVIDIA 드라이버가 설치되지 않았습니다.")
            return False
        except Exception as e:
            print_error(f"nvidia-smi 확인 중 오류: {str(e)}")
            return False

def check_pytorch():
    """PyTorch CUDA 지원 확인"""
    print_header("3. PyTorch CUDA 지원 확인")
    try:
        import torch
        print_success(f"PyTorch 버전: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print_success(f"PyTorch CUDA 사용 가능: {cuda_available}")
            print_info(f"CUDA 버전 (PyTorch): {torch.version.cuda}")
            print_info(f"GPU 개수: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print_info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print_info(f"  - 메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            
            # 간단한 GPU 연산 테스트
            print("\n[GPU 연산 테스트]")
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                z = torch.matmul(x, y)
                print_success("GPU 행렬 연산 테스트 성공!")
                print_info(f"결과 텐서 shape: {z.shape}, device: {z.device}")
            except Exception as e:
                print_error(f"GPU 연산 테스트 실패: {str(e)}")
        else:
            print_error("PyTorch에서 CUDA를 사용할 수 없습니다.")
            print_info("CPU 버전의 PyTorch가 설치되어 있을 수 있습니다.")
        
        return cuda_available
    except ImportError:
        print_error("PyTorch가 설치되지 않았습니다.")
        print_info("설치: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    except Exception as e:
        print_error(f"PyTorch 확인 중 오류: {str(e)}")
        return False

def check_tensorflow():
    """TensorFlow CUDA 지원 확인"""
    print_header("4. TensorFlow CUDA 지원 확인")
    try:
        import tensorflow as tf
        print_success(f"TensorFlow 버전: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print_success(f"TensorFlow에서 GPU {len(gpus)}개 감지")
            for i, gpu in enumerate(gpus):
                print_info(f"GPU {i}: {gpu.name}")
                
            # GPU 메모리 성장 설정
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print_success("GPU 메모리 동적 할당 설정 완료")
            except RuntimeError as e:
                print_error(f"메모리 설정 실패: {str(e)}")
            
            # 간단한 GPU 연산 테스트
            print("\n[GPU 연산 테스트]")
            try:
                with tf.device('/GPU:0'):
                    a = tf.random.normal([1000, 1000])
                    b = tf.random.normal([1000, 1000])
                    c = tf.matmul(a, b)
                print_success("GPU 행렬 연산 테스트 성공!")
                print_info(f"결과 텐서 shape: {c.shape}")
            except Exception as e:
                print_error(f"GPU 연산 테스트 실패: {str(e)}")
            
            return True
        else:
            print_error("TensorFlow에서 GPU를 감지하지 못했습니다.")
            return False
    except ImportError:
        print_error("TensorFlow가 설치되지 않았습니다.")
        print_info("설치: pip install tensorflow[and-cuda]")
        return False
    except Exception as e:
        print_error(f"TensorFlow 확인 중 오류: {str(e)}")
        return False

def check_xgboost():
    """XGBoost GPU 지원 확인"""
    print_header("5. XGBoost GPU 지원 확인")
    try:
        import xgboost as xgb
        print_success(f"XGBoost 버전: {xgb.__version__}")
        
        # XGBoost GPU 빌드 확인
        try:
            # 간단한 GPU 테스트
            import numpy as np
            from sklearn.datasets import make_classification
            
            X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
            dtrain = xgb.DMatrix(X, label=y)
            
            params = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0
            }
            
            print_info("GPU 학습 테스트 중...")
            bst = xgb.train(params, dtrain, num_boost_round=10)
            print_success("XGBoost GPU 학습 테스트 성공!")
            return True
            
        except Exception as e:
            print_error(f"XGBoost GPU 테스트 실패: {str(e)}")
            print_info("GPU 빌드가 아니거나 CUDA 설정에 문제가 있을 수 있습니다.")
            return False
            
    except ImportError:
        print_error("XGBoost가 설치되지 않았습니다.")
        return False
    except Exception as e:
        print_error(f"XGBoost 확인 중 오류: {str(e)}")
        return False

def check_cupy():
    """CuPy CUDA 지원 확인"""
    print_header("6. CuPy CUDA 지원 확인")
    try:
        import cupy as cp
        print_success(f"CuPy 버전: {cp.__version__}")
        
        # GPU 정보 출력
        print_info(f"사용 가능한 메모리: {cp.cuda.Device().mem_info[0] / 1024**3:.2f} GB")
        print_info(f"전체 메모리: {cp.cuda.Device().mem_info[1] / 1024**3:.2f} GB")
        
        # 간단한 연산 테스트
        try:
            x = cp.random.randn(1000, 1000)
            y = cp.random.randn(1000, 1000)
            z = cp.matmul(x, y)
            print_success("CuPy GPU 연산 테스트 성공!")
        except Exception as e:
            print_error(f"CuPy 연산 테스트 실패: {str(e)}")
            return False
        
        return True
    except ImportError:
        print_error("CuPy가 설치되지 않았습니다.")
        print_info("설치: pip install cupy-cuda11x (CUDA 버전에 맞게)")
        return False
    except Exception as e:
        print_error(f"CuPy 확인 중 오류: {str(e)}")
        return False

def print_summary(results):
    """결과 요약 출력"""
    print_header("종합 진단 결과")
    
    status_items = [
        ("NVIDIA 드라이버", results.get('nvidia_smi', False)),
        ("CUDA Toolkit", results.get('cuda_toolkit', False)),
        ("PyTorch CUDA", results.get('pytorch', False)),
        ("TensorFlow GPU", results.get('tensorflow', False)),
        ("XGBoost GPU", results.get('xgboost', False)),
        ("CuPy", results.get('cupy', False)),
    ]
    
    for item, status in status_items:
        if status:
            print_success(f"{item}: 사용 가능")
        else:
            print_error(f"{item}: 사용 불가")
    
    print("\n" + "="*70)
    
    # 권장 사항
    if not results.get('nvidia_smi', False):
        print("\n[WARNING] NVIDIA 드라이버가 감지되지 않습니다.")
        print("   -> https://www.nvidia.com/Download/index.aspx 에서 드라이버 설치")
    
    if not results.get('cuda_toolkit', False):
        print("\n[WARNING] CUDA Toolkit이 감지되지 않습니다.")
        print("   -> https://developer.nvidia.com/cuda-downloads 에서 CUDA 설치")
    
    if results.get('nvidia_smi', False) and not results.get('pytorch', False):
        print("\n[TIP] PyTorch GPU 버전 설치 권장:")
        print("   -> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    if results.get('nvidia_smi', False) and not results.get('xgboost', False):
        print("\n[TIP] XGBoost GPU 버전이 제대로 작동하지 않습니다.")
        print("   현재 프로젝트에서 사용 중인 XGBoost는 CPU 버전일 수 있습니다.")

def main():
    """메인 함수"""
    print("\n" + "="*70)
    print("  GPU 및 CUDA 설정 종합 진단 도구")
    print("="*70)
    print_info(f"Python 버전: {sys.version}")
    print_info(f"플랫폼: {platform.platform()}")
    
    results = {}
    
    # 각 항목 확인
    results['nvidia_smi'] = check_nvidia_smi()
    results['cuda_toolkit'] = check_cuda_toolkit()
    results['pytorch'] = check_pytorch()
    results['tensorflow'] = check_tensorflow()
    results['xgboost'] = check_xgboost()
    results['cupy'] = check_cupy()
    
    # 결과 요약
    print_summary(results)

if __name__ == "__main__":
    main()


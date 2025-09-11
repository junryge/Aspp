#!/usr/bin/env python3
"""
NVIDIA V100 GPU Test Script - TensorFlow Version
GPU 인식, 메모리, 연산 성능 등을 종합적으로 테스트
"""

import subprocess
import sys
import os
import time

def check_nvidia_smi():
    """nvidia-smi로 GPU 정보 확인"""
    print("="*60)
    print("1. NVIDIA-SMI GPU 정보")
    print("="*60)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("❌ nvidia-smi를 찾을 수 없습니다. NVIDIA 드라이버가 설치되어 있는지 확인하세요.")
        return False
    return True

def check_cuda():
    """CUDA 설치 확인"""
    print("="*60)
    print("2. CUDA 버전 확인")
    print("="*60)
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("⚠️  nvcc를 찾을 수 없습니다. CUDA toolkit이 설치되어 있지 않을 수 있습니다.")
        print("   TensorFlow는 자체 CUDA 런타임을 포함하므로 계속 진행합니다.")
    print()

def test_tensorflow():
    """TensorFlow로 GPU 테스트"""
    print("="*60)
    print("3. TensorFlow GPU 테스트")
    print("="*60)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow 버전: {tf.__version__}")
        
        # GPU 목록 확인
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✅ 감지된 GPU 개수: {len(gpus)}")
        
        if len(gpus) > 0:
            print("\n📋 GPU 상세 정보:")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
                
                # GPU 메모리 증가 허용 (필요시에만 메모리 할당)
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"   ⚠️  메모리 증가 설정 실패: {e}")
            
            # CUDA 및 cuDNN 정보
            print(f"\n✅ CUDA 지원: {tf.test.is_built_with_cuda()}")
            print(f"✅ GPU 가속 지원: {tf.test.is_built_with_gpu_support()}")
            
            # 각 GPU에서 간단한 연산 테스트
            print("\n📊 각 GPU별 연산 테스트:")
            for i in range(len(gpus)):
                with tf.device(f'/GPU:{i}'):
                    print(f"\n   GPU {i} 테스트:")
                    
                    # 작은 행렬로 시작
                    size = 1000
                    a = tf.random.normal([size, size])
                    b = tf.random.normal([size, size])
                    
                    # Warm-up
                    c = tf.matmul(a, b)
                    
                    # 실제 측정
                    start = time.time()
                    for _ in range(10):
                        c = tf.matmul(a, b)
                    tf.debugging.assert_all_finite(c, "결과 확인")
                    elapsed = time.time() - start
                    
                    print(f"   - 1000x1000 행렬곱 10회: {elapsed:.3f}초")
                    print(f"   - 처리량: {10 * 2 * size**3 / elapsed / 1e9:.1f} GFLOPS")
            
            # 큰 행렬 연산 테스트 (메모리 테스트)
            print("\n📊 대용량 행렬 연산 테스트 (GPU 0):")
            with tf.device('/GPU:0'):
                try:
                    size = 10000
                    print(f"   {size}x{size} 행렬 생성 중...")
                    a = tf.random.normal([size, size], dtype=tf.float32)
                    b = tf.random.normal([size, size], dtype=tf.float32)
                    
                    print(f"   행렬곱 연산 중...")
                    start = time.time()
                    c = tf.matmul(a, b)
                    tf.debugging.assert_all_finite(c, "결과 확인")
                    elapsed = time.time() - start
                    
                    memory_gb = (3 * size * size * 4) / (1024**3)  # 3개 행렬, float32(4bytes)
                    print(f"   ✅ 성공! 시간: {elapsed:.3f}초")
                    print(f"   - 대략적인 메모리 사용량: {memory_gb:.2f} GB")
                    print(f"   - 처리량: {2 * size**3 / elapsed / 1e12:.2f} TFLOPS")
                    
                except tf.errors.ResourceExhaustedError:
                    print("   ❌ 메모리 부족! 더 작은 크기로 시도하세요.")
                except Exception as e:
                    print(f"   ❌ 오류 발생: {e}")
            
            # Multi-GPU 전략 테스트
            if len(gpus) > 1:
                print("\n📊 Multi-GPU 전략 테스트:")
                strategy = tf.distribute.MirroredStrategy()
                print(f"   사용 가능한 디바이스: {strategy.num_replicas_in_sync}개")
                
                with strategy.scope():
                    # 간단한 모델 생성
                    model = tf.keras.Sequential([
                        tf.keras.layers.Dense(1000, activation='relu', input_shape=(1000,)),
                        tf.keras.layers.Dense(1000, activation='relu'),
                        tf.keras.layers.Dense(10, activation='softmax')
                    ])
                    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
                    print("   ✅ Multi-GPU 모델 생성 성공!")
                    
                    # 모델 요약
                    print("\n   모델 구조:")
                    model.summary()
            
            # GPU 간 통신 테스트
            if len(gpus) > 1:
                print("\n📊 GPU 간 데이터 전송 테스트:")
                size = 1000
                with tf.device('/GPU:0'):
                    gpu0_data = tf.random.normal([size, size])
                
                start = time.time()
                with tf.device('/GPU:1'):
                    gpu1_data = tf.identity(gpu0_data)  # GPU 0 -> GPU 1 복사
                    result = tf.matmul(gpu1_data, gpu1_data)
                elapsed = time.time() - start
                
                data_size_mb = (size * size * 4) / (1024**2)
                print(f"   GPU 0 → GPU 1 전송 및 연산")
                print(f"   - 데이터 크기: {data_size_mb:.2f} MB")
                print(f"   - 전송 + 연산 시간: {elapsed:.3f}초")
                print(f"   - 대역폭: {data_size_mb / elapsed:.1f} MB/s")
        
        else:
            print("❌ GPU를 찾을 수 없습니다!")
            print("   다음을 확인하세요:")
            print("   1. NVIDIA 드라이버가 설치되어 있는지")
            print("   2. tensorflow-gpu가 설치되어 있는지 (pip install tensorflow-gpu)")
            print("   3. CUDA와 cuDNN이 올바르게 설치되어 있는지")
            
    except ImportError:
        print("❌ TensorFlow가 설치되어 있지 않습니다!")
        print("   설치 명령어: pip install tensorflow")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """메인 함수"""
    print("\n" + "="*60)
    print(" NVIDIA V100 GPU 테스트 스크립트 (TensorFlow)")
    print("="*60 + "\n")
    
    # 환경 변수 설정 (선택사항)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # INFO 메시지 숨김
    
    # 테스트 실행
    if check_nvidia_smi():
        check_cuda()
        if test_tensorflow():
            print("\n" + "="*60)
            print("✅ 모든 테스트 완료!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("⚠️  일부 테스트 실패")
            print("="*60)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
간단한 TensorFlow GPU 확인 스크립트
빠르게 GPU가 인식되는지만 확인
"""

import tensorflow as tf

print("TensorFlow 버전:", tf.__version__)
print("\n사용 가능한 GPU:")
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
    
    # 간단한 연산 테스트
    print("\n간단한 GPU 연산 테스트:")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print("GPU 0에서 행렬곱 결과:")
        print(c.numpy())
    
    if len(gpus) > 1:
        with tf.device('/GPU:1'):
            d = tf.matmul(a, b)
            print("\nGPU 1에서 행렬곱 결과:")
            print(d.numpy())
else:
    print("GPU를 찾을 수 없습니다!")
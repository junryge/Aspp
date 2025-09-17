"""
🚀 반도체 물류 TOTALCNT 전체 구간 예측 시스템
=============================================
목표: 100분 과거 데이터로 10분 후 TOTALCNT 예측

예측 구간:
- 정상: 0-1399 (56.53%)
- 주의: 1400-1699 (43.12%)  
- 심각: 1700+ (0.35%) ← 핵심 목표

5개 딥러닝 모델:
1. StableLSTM - 전체 시계열 패턴
2. PatchTST - Transformer 기반 패치 학습
3. SpikeDetector - 급변 감지
4. ExtremeNet - 극단값 전문
5. EnsembleBase - 앙상블 통합
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# 설정
tf.random.set_seed(42)
np.random.seed(42)

print("="*80)
print("🚀 반도체 물류 예측 시스템 v12.0")
print(f"📦 TensorFlow: {tf.__version__}")
print("="*80)
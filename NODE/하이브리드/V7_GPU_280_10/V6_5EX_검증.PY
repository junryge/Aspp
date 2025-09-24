"""
🔍 EXTREME_NET 학습 전 최종 검증
=====================================
학습 시작 전 데이터와 환경 체크
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import os
import psutil
import gc

print("="*80)
print("🔍 학습 전 최종 검증 시작")
print("="*80)

# 1. TensorFlow 버전 확인
print("\n✅ 1. TensorFlow 환경")
print(f"  TensorFlow 버전: {tf.__version__}")
if tf.__version__.startswith('2.16'):
    print("  ✓ TensorFlow 2.16.x 확인")
else:
    print(f"  ⚠️ 주의: TensorFlow 2.16.1 권장 (현재: {tf.__version__})")

# 2. GPU 확인
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"\n✅ 2. GPU 환경")
    print(f"  GPU 개수: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
else:
    print("\n⚠️ 2. CPU 모드로 실행 (학습 속도 느림)")

# 3. 메모리 확인
print(f"\n✅ 3. 시스템 메모리")
memory = psutil.virtual_memory()
print(f"  전체 메모리: {memory.total / (1024**3):.1f} GB")
print(f"  사용 가능: {memory.available / (1024**3):.1f} GB")
print(f"  사용률: {memory.percent:.1f}%")

if memory.available / (1024**3) < 4:
    print("  ⚠️ 메모리 부족 가능성 (최소 4GB 권장)")
else:
    print("  ✓ 충분한 메모리")

# 4. 데이터 파일 확인
print(f"\n✅ 4. 데이터 파일 확인")
data_paths = [
    '/mnt/user-data/uploads/gs.CSV',
    'uu.csv',
    'uu2.csv',
    'data.csv'
]

found_files = []
for path in data_paths:
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024**2)  # MB
        print(f"  ✓ {path} ({size:.1f} MB)")
        found_files.append(path)
    else:
        print(f"  ✗ {path} (없음)")

if not found_files:
    print("\n❌ 데이터 파일 없음! 학습 불가")
    exit(1)

# 5. 데이터 검증
print(f"\n✅ 5. 데이터 검증")
data_file = found_files[0]
print(f"  선택된 파일: {data_file}")

try:
    df = pd.read_csv(data_file)
    print(f"  데이터 크기: {len(df):,} 행 × {len(df.columns)} 열")
    
    # 필수 컬럼 확인
    required_cols = ['TOTALCNT', 'M14AM14B', 'M14AM10A']
    missing_cols = []
    
    print("\n  필수 컬럼 확인:")
    for col in required_cols:
        if col in df.columns:
            print(f"    ✓ {col}")
        else:
            print(f"    ✗ {col} (누락)")
            missing_cols.append(col)
    
    if missing_cols:
        print(f"\n❌ 필수 컬럼 누락: {missing_cols}")
        print("   다음 컬럼이 있습니다:", list(df.columns[:10]))
        exit(1)
    
    # 데이터 통계
    print("\n  TOTALCNT 통계:")
    print(f"    최소: {df['TOTALCNT'].min():.0f}")
    print(f"    최대: {df['TOTALCNT'].max():.0f}")
    print(f"    평균: {df['TOTALCNT'].mean():.0f}")
    print(f"    표준편차: {df['TOTALCNT'].std():.0f}")
    
    # 0값 체크
    zero_count = (df['TOTALCNT'] == 0).sum()
    if zero_count > 0:
        print(f"    ⚠️ 0값 {zero_count}개 (제거됨)")
    
    # NaN 체크
    nan_count = df['TOTALCNT'].isna().sum()
    if nan_count > 0:
        print(f"    ⚠️ NaN값 {nan_count}개")
    
    # 시퀀스 생성 가능 여부
    min_required = 100 + 10  # seq_len + pred_len
    if len(df) < min_required:
        print(f"\n❌ 데이터 부족: 최소 {min_required}개 필요 (현재: {len(df)})")
        exit(1)
    
    # 예상 시퀀스 수
    expected_sequences = len(df) - min_required + 1
    print(f"\n  예상 시퀀스 수: {expected_sequences:,}")
    print(f"    Train: ~{int(expected_sequences * 0.7):,}")
    print(f"    Val: ~{int(expected_sequences * 0.15):,}")
    print(f"    Test: ~{int(expected_sequences * 0.15):,}")
    
    # UU1/UU2 패턴 체크
    print("\n  패턴 분석:")
    m14b_mean = df['M14AM14B'].mean()
    high_count = (df['TOTALCNT'] > 1682).sum()
    m14b_350_ratio = (df['M14AM14B'] > 350).mean()
    
    if high_count > 5 or m14b_mean > 380 or m14b_350_ratio > 0.3:
        print(f"    패턴: UU2 (고값 유지)")
    else:
        print(f"    패턴: UU1 (급증 가능)")
    
    print(f"    M14B 평균: {m14b_mean:.0f}")
    print(f"    1682+ 케이스: {high_count}개")
    print(f"    M14B>350 비율: {m14b_350_ratio:.1%}")
    
except Exception as e:
    print(f"\n❌ 데이터 로드 실패: {e}")
    exit(1)

# 6. 예상 학습 시간
print(f"\n✅ 6. 예상 학습 시간")
if gpus:
    est_time = expected_sequences / 10000 * 5  # GPU 기준
    print(f"  GPU 모드: 약 {est_time:.0f}~{est_time*2:.0f}분")
else:
    est_time = expected_sequences / 10000 * 30  # CPU 기준
    print(f"  CPU 모드: 약 {est_time:.0f}~{est_time*2:.0f}분")

# 7. 최종 체크리스트
print(f"\n✅ 7. 최종 체크리스트")
checklist = {
    "TensorFlow 설치": tf.__version__ is not None,
    "데이터 파일": len(found_files) > 0,
    "필수 컬럼": len(missing_cols) == 0 if 'missing_cols' in locals() else False,
    "충분한 데이터": expected_sequences > 1000 if 'expected_sequences' in locals() else False,
    "메모리": memory.available / (1024**3) > 4
}

all_ready = True
for item, status in checklist.items():
    status_text = "✓" if status else "✗"
    print(f"  {status_text} {item}")
    if not status:
        all_ready = False

# 8. 결과
print("\n" + "="*80)
if all_ready:
    print("✅ 모든 준비 완료! 학습을 시작할 수 있습니다.")
    print("\n실행 명령:")
    print("  python extreme_net_complete_fixed.py")
    print("\n예상 소요 시간: ", end="")
    if gpus:
        print(f"{est_time:.0f}~{est_time*2:.0f}분 (GPU)")
    else:
        print(f"{est_time:.0f}~{est_time*2:.0f}분 (CPU)")
else:
    print("❌ 문제가 있습니다. 위의 체크리스트를 확인하세요.")

print("="*80)

# 메모리 정리
gc.collect()
if gpus:
    tf.keras.backend.clear_session()
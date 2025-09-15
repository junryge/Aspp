"""
scaler_from_sequence.py
이미 있는 시퀀스 파일(sequences_v6.npz)에서 스케일러만 생성
CPU 버전 - 빠르게 실행
"""

import numpy as np
from sklearn.preprocessing import RobustScaler
import pickle
import os
from datetime import datetime

print("="*60)
print("🔧 시퀀스에서 스케일러 추출/생성 프로그램")
print("="*60)

# ============================================
# 설정
# ============================================
class Config:
    # 기존 시퀀스 파일 경로 (이미 있는 파일)
    SEQUENCE_FILE = './sequences_v6.npz'  # 👈 여기에 실제 시퀀스 파일 경로
    
    # 생성할 스케일러 파일 경로
    SCALER_FILE = './scalers_v6_gpu.pkl'  # 👈 저장할 스케일러 파일명
    
    # 스케일 방식
    USE_ROBUST = True  # True: RobustScaler, False: StandardScaler

# ============================================
# 메인 함수
# ============================================
def create_scaler_from_sequence():
    """기존 시퀀스에서 스케일러 생성"""
    
    print(f"\n📂 시퀀스 파일 로드 중: {Config.SEQUENCE_FILE}")
    
    # 1. 시퀀스 파일 확인
    if not os.path.exists(Config.SEQUENCE_FILE):
        print(f"❌ 시퀀스 파일이 없습니다: {Config.SEQUENCE_FILE}")
        print("파일 경로를 확인해주세요!")
        return False
    
    # 2. 시퀀스 로드
    try:
        data = np.load(Config.SEQUENCE_FILE)
        print("✅ 시퀀스 로드 성공")
        
        # 데이터 확인
        print("\n📊 로드된 데이터:")
        for key in data.files:
            if isinstance(data[key], np.ndarray):
                print(f"  - {key}: shape {data[key].shape}, dtype {data[key].dtype}")
            else:
                print(f"  - {key}: {type(data[key])}")
    except Exception as e:
        print(f"❌ 시퀀스 로드 실패: {e}")
        return False
    
    # 3. 필수 데이터 추출
    print("\n📦 데이터 추출 중...")
    
    # X 데이터 (이미 스케일된 데이터일 수도 있음)
    if 'X' in data.files:
        X = data['X']
        print(f"  ✅ X 데이터: {X.shape}")
    else:
        print("❌ X 데이터가 없습니다!")
        return False
    
    # y 데이터 (원본 또는 스케일된 버전)
    y_original = None
    y_scaled = None
    
    if 'y_original' in data.files:
        y_original = data['y_original']
        print(f"  ✅ y_original: {y_original.shape}")
    elif 'y' in data.files:
        y_original = data['y']
        print(f"  ✅ y: {y_original.shape}")
    
    if 'y_scaled' in data.files:
        y_scaled = data['y_scaled']
        print(f"  ✅ y_scaled: {y_scaled.shape}")
    
    # M14 특징
    m14_features = None
    m14_features_scaled = None
    
    if 'm14_features' in data.files:
        m14_features = data['m14_features']
        print(f"  ✅ m14_features: {m14_features.shape}")
    
    if 'm14_features_scaled' in data.files:
        m14_features_scaled = data['m14_features_scaled']
        print(f"  ✅ m14_features_scaled: {m14_features_scaled.shape}")
    
    # 특징 이름 (있으면)
    feature_names = None
    if 'feature_names' in data.files:
        feature_names = data['feature_names']
        print(f"  ✅ feature_names: {len(feature_names)}개")
    
    # ============================================
    # 4. 스케일러 생성
    # ============================================
    print("\n📏 스케일러 생성 중...")
    
    # 스케일러 타입 선택
    if Config.USE_ROBUST:
        ScalerClass = RobustScaler
        print("  스케일러 타입: RobustScaler (이상치에 강함)")
    else:
        from sklearn.preprocessing import StandardScaler
        ScalerClass = StandardScaler
        print("  스케일러 타입: StandardScaler")
    
    # 1. X 특징 스케일러
    print("\n  1️⃣ X 특징 스케일러 생성...")
    feature_scalers = {}
    
    n_features = X.shape[2]
    print(f"    특징 수: {n_features}개")
    
    for i in range(n_features):
        # 각 특징별로 스케일러 생성
        scaler = ScalerClass()
        
        # 2D로 reshape하여 fit
        feature_data = X[:, :, i].reshape(-1, 1)
        scaler.fit(feature_data)
        
        # 저장
        feature_scalers[f'feature_{i}'] = scaler
        
        # 진행 표시
        if (i + 1) % 10 == 0 or i == n_features - 1:
            print(f"      {i + 1}/{n_features} 완료")
    
    print(f"  ✅ {len(feature_scalers)}개 특징 스케일러 생성 완료")
    
    # 2. y 스케일러
    y_scaler = None
    if y_original is not None:
        print("\n  2️⃣ y(타겟) 스케일러 생성...")
        y_scaler = ScalerClass()
        y_scaler.fit(y_original.reshape(-1, 1))
        
        print(f"  ✅ y 스케일러 생성 완료")
        print(f"    원본 범위: {y_original.min():.0f} ~ {y_original.max():.0f}")
        
        # 검증: 스케일 테스트
        y_test = y_scaler.transform(y_original[:10].reshape(-1, 1))
        print(f"    테스트 스케일 범위: {y_test.min():.2f} ~ {y_test.max():.2f}")
    
    # 3. M14 스케일러
    m14_scaler = None
    if m14_features is not None:
        print("\n  3️⃣ M14 특징 스케일러 생성...")
        m14_scaler = ScalerClass()
        m14_scaler.fit(m14_features)
        
        print(f"  ✅ M14 스케일러 생성 완료")
        print(f"    M14 shape: {m14_features.shape}")
        print(f"    M14B 범위: {m14_features[:, 0].min():.0f} ~ {m14_features[:, 0].max():.0f}")
        print(f"    M10A 범위: {m14_features[:, 1].min():.0f} ~ {m14_features[:, 1].max():.0f}")
    
    # ============================================
    # 5. 스케일러 저장
    # ============================================
    print("\n💾 스케일러 저장 중...")
    
    # 저장할 딕셔너리
    scaler_dict = {
        'feature_scalers': feature_scalers,
        'y_scaler': y_scaler,
        'm14_scaler': m14_scaler,
        'creation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'scaler_type': 'RobustScaler' if Config.USE_ROBUST else 'StandardScaler',
        'source_file': Config.SEQUENCE_FILE,
        'data_info': {
            'n_samples': X.shape[0],
            'n_timesteps': X.shape[1],
            'n_features': X.shape[2],
        }
    }
    
    # 특징 이름 추가 (있으면)
    if feature_names is not None:
        scaler_dict['feature_names'] = feature_names
    
    # y 범위 정보 추가
    if y_original is not None:
        scaler_dict['data_info']['y_range'] = [float(y_original.min()), float(y_original.max())]
    
    # pkl 파일로 저장
    try:
        with open(Config.SCALER_FILE, 'wb') as f:
            pickle.dump(scaler_dict, f)
        
        print(f"  ✅ 스케일러 저장 완료: {Config.SCALER_FILE}")
        
        # 파일 크기 확인
        file_size = os.path.getsize(Config.SCALER_FILE) / 1024  # KB
        print(f"  📁 파일 크기: {file_size:.1f} KB")
    except Exception as e:
        print(f"❌ 저장 실패: {e}")
        return False
    
    # ============================================
    # 6. 검증 (다시 로드해서 테스트)
    # ============================================
    print("\n🔍 스케일러 검증 중...")
    
    try:
        # 저장된 스케일러 로드
        with open(Config.SCALER_FILE, 'rb') as f:
            loaded_scalers = pickle.load(f)
        
        print("  ✅ 스케일러 로드 성공")
        print(f"    특징 스케일러 수: {len(loaded_scalers['feature_scalers'])}")
        print(f"    스케일러 타입: {loaded_scalers['scaler_type']}")
        print(f"    생성 시간: {loaded_scalers['creation_time']}")
        print(f"    소스 파일: {loaded_scalers['source_file']}")
        
        # 샘플 변환 테스트
        print("\n  📊 샘플 변환 테스트:")
        
        # X 테스트
        test_feature = X[0, :, 0].reshape(-1, 1)  # 첫 번째 샘플의 첫 번째 특징
        test_scaler = loaded_scalers['feature_scalers']['feature_0']
        test_scaled = test_scaler.transform(test_feature[:5].reshape(-1, 1))
        print(f"    X 원본 샘플: {test_feature[:5].flatten()}")
        print(f"    X 스케일 샘플: {test_scaled.flatten()}")
        
        # y 테스트
        if loaded_scalers['y_scaler'] is not None and y_original is not None:
            test_y = y_original[:5].reshape(-1, 1)
            test_y_scaled = loaded_scalers['y_scaler'].transform(test_y)
            print(f"    y 원본 샘플: {test_y.flatten()}")
            print(f"    y 스케일 샘플: {test_y_scaled.flatten()}")
            
            # 역변환 테스트
            test_y_inverse = loaded_scalers['y_scaler'].inverse_transform(test_y_scaled)
            print(f"    y 역변환 샘플: {test_y_inverse.flatten()}")
            print(f"    역변환 오차: {np.abs(test_y.flatten() - test_y_inverse.flatten()).mean():.6f}")
        
        print("\n✅ 모든 검증 통과!")
        
    except Exception as e:
        print(f"❌ 검증 실패: {e}")
        return False
    
    # ============================================
    # 7. 완료
    # ============================================
    print("\n" + "="*60)
    print("🎉 스케일러 생성 완료!")
    print("="*60)
    print(f"📁 스케일러 파일: {Config.SCALER_FILE}")
    print(f"📊 특징 수: {n_features}개")
    if y_original is not None:
        print(f"📊 y 범위: {y_original.min():.0f} ~ {y_original.max():.0f}")
    if m14_features is not None:
        print(f"📊 M14 특징: {m14_features.shape[1]}개")
    print("\n💡 이제 학습 코드에서 이 스케일러를 사용할 수 있습니다!")
    
    return True

# ============================================
# 실행
# ============================================
if __name__ == "__main__":
    success = create_scaler_from_sequence()
    
    if success:
        print("\n✅ 프로그램 정상 종료")
    else:
        print("\n❌ 프로그램 실패")
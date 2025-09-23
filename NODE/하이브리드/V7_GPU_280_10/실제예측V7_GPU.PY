# -*- coding: utf-8 -*-
"""
ExtraTrees 실시간 예측 시스템
280분 데이터로 10분 후 예측
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔥 ExtraTrees 실시간 예측 시스템")
print("="*80)

# 설정
MODEL_DIR = 'regression_classification_models_280to10'
DATA_FILE = 'data/20250807_DATA.csv'

def create_features_from_sequence(data, seq_length=280):
    """
    280분 시퀀스 데이터에서 특징 추출
    """
    if len(data) < seq_length:
        print(f"⚠️ 데이터가 부족합니다. 필요: {seq_length}개, 현재: {len(data)}개")
        return None
    
    # 마지막 280분 데이터만 사용
    seq_data = data.iloc[-seq_length:].copy()
    
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 'M16M14A', 'TOTALCNT']
    
    # 파생 변수 생성
    seq_data['ratio_M14B_M14A'] = np.clip(seq_data['M14AM14B'] / (seq_data['M14AM10A'] + 1), 0, 1000)
    seq_data['ratio_M14B_M16'] = np.clip(seq_data['M14AM14B'] / (seq_data['M14AM16'] + 1), 0, 1000)
    seq_data['totalcnt_change'] = seq_data['TOTALCNT'].diff().fillna(0)
    seq_data['totalcnt_pct_change'] = np.clip(seq_data['TOTALCNT'].pct_change().fillna(0), -10, 10)
    
    seq_data = seq_data.replace([np.inf, -np.inf], 0).fillna(0)
    
    features = []
    
    # 각 컬럼별 특징 추출
    for col in feature_cols:
        values = seq_data[col].values
        
        # 기본 통계
        features.extend([
            np.mean(values),
            np.std(values) if len(values) > 1 else 0,
            np.min(values),
            np.max(values),
            np.percentile(values, 25),
            np.percentile(values, 50),
            np.percentile(values, 75),
            values[-1],  # 현재값
            values[-1] - values[0],  # 전체 변화량
            np.mean(values[-60:]),  # 최근 1시간 평균
            np.max(values[-60:]),   # 최근 1시간 최대
            np.mean(values[-30:]),  # 최근 30분 평균
            np.max(values[-30:]),   # 최근 30분 최대
        ])
        
        # TOTALCNT 특별 처리
        if col == 'TOTALCNT':
            features.append(np.sum((values >= 1650) & (values < 1700)))
            features.append(np.sum(values >= 1700))
            features.append(np.max(values[-20:]))
            features.append(np.sum(values < 1400))
            features.append(np.sum((values >= 1400) & (values < 1700)))
            features.append(np.sum(values >= 1700))
            features.append(np.sum((values >= 1651) & (values <= 1682)))
            
            anomaly_values = values[(values >= 1651) & (values <= 1682)]
            features.append(np.max(anomaly_values) if len(anomaly_values) > 0 else 0)
            
            normal_vals = values[values < 1400]
            check_vals = values[(values >= 1400) & (values < 1700)]
            danger_vals = values[values >= 1700]
            
            features.append(np.mean(normal_vals) if len(normal_vals) > 0 else 0)
            features.append(np.mean(check_vals) if len(check_vals) > 0 else 0)
            features.append(np.mean(danger_vals) if len(danger_vals) > 0 else 0)
            
            try:
                x = np.arange(len(values))
                slope, _ = np.polyfit(x, values, 1)
                features.append(np.clip(slope, -100, 100))
            except:
                features.append(0)
            
            try:
                recent_slope = np.polyfit(np.arange(60), values[-60:], 1)[0]
                features.append(np.clip(recent_slope, -100, 100))
            except:
                features.append(0)
    
    # 마지막 시점 파생 변수
    features.extend([
        np.clip(seq_data['ratio_M14B_M14A'].iloc[-1], 0, 1000),
        np.clip(seq_data['ratio_M14B_M16'].iloc[-1], 0, 1000),
        np.clip(seq_data['totalcnt_change'].iloc[-1], -1000, 1000),
        np.clip(seq_data['totalcnt_pct_change'].iloc[-1], -10, 10),
    ])
    
    return np.array(features)

def get_level_info(predicted_value):
    """
    예측값에 따른 구간 정보 반환
    """
    if predicted_value < 1400:
        return "정상", 100, 0, 0, "🟢"
    elif predicted_value < 1700:
        return "주의", 0, 100, 0, "🟡"
    else:
        # 1700 이상일 때 세부 확률 계산
        if predicted_value >= 1800:
            return "심각", 0, 5, 95, "🔴"
        elif predicted_value >= 1750:
            return "심각", 0, 20, 80, "🔴"
        else:
            return "심각", 0, 40, 60, "🔴"

def format_time(time_str):
    """
    시간 형식 변환 (202507310249 -> 2025-07-31 02:49)
    """
    if len(time_str) == 12:
        year = time_str[:4]
        month = time_str[4:6]
        day = time_str[6:8]
        hour = time_str[8:10]
        minute = time_str[10:12]
        return f"{year}-{month}-{day} {hour}:{minute}"
    return time_str

def predict_next_10min(data_file, model_dir):
    """
    실시간 예측 실행
    """
    # 1. 데이터 로드
    print("데이터 로딩 중...")
    df = pd.read_csv(data_file)
    print(f"✓ 데이터 크기: {len(df)}행")
    
    # TIME 컬럼을 문자열로 변환
    if 'TIME' in df.columns:
        df['TIME'] = df['TIME'].astype(str)
    elif 'CURRTIME' in df.columns:
        df['TIME'] = df['CURRTIME'].astype(str)
    
    # 최소 280개 데이터 필요
    if len(df) < 280:
        print(f"❌ 데이터 부족! 최소 280개 필요 (현재: {len(df)}개)")
        return None
    
    # 2. 특징 추출
    print("특징 추출 중...")
    features = create_features_from_sequence(df)
    
    if features is None:
        return None
    
    # 3. 모델 로드
    print("모델 로딩 중...")
    model_path = os.path.join(model_dir, 'ExtraTrees_regression_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return None
    
    if not os.path.exists(scaler_path):
        print(f"❌ 스케일러 파일을 찾을 수 없습니다: {scaler_path}")
        return None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # 4. 특징 정규화
    features_2d = features.reshape(1, -1)
    
    # 특징 개수 맞추기
    expected_features = scaler.n_features_in_
    if features_2d.shape[1] < expected_features:
        features_2d = np.hstack([features_2d, np.zeros((1, expected_features - features_2d.shape[1]))])
    elif features_2d.shape[1] > expected_features:
        features_2d = features_2d[:, :expected_features]
    
    features_scaled = scaler.transform(features_2d)
    
    # 5. 예측
    print("예측 중...")
    prediction = model.predict(features_scaled)[0]
    
    # 6. 보정 (1700+ 구간 강화)
    current_totalcnt = df['TOTALCNT'].iloc[-1]
    seq_max = df['TOTALCNT'].iloc[-280:].max()
    
    # 보정 로직
    if current_totalcnt >= 1700 and prediction < 1700:
        prediction = prediction * 1.08  # 8% 상향
    elif current_totalcnt >= 1650 and prediction < current_totalcnt:
        prediction = prediction * 1.05  # 5% 상향
    elif seq_max >= 1700 and prediction < 1650:
        prediction = prediction * 1.06  # 6% 상향
    
    # 7. 결과 생성
    prediction = round(prediction)
    level, normal_prob, warning_prob, danger_prob, emoji = get_level_info(prediction)
    
    # 현재 시간과 예측 시간
    current_time_str = df['TIME'].iloc[-1]
    current_time_formatted = format_time(current_time_str)
    
    # 10분 후 시간 계산
    try:
        dt = datetime.strptime(current_time_str, "%Y%m%d%H%M")
        pred_time = dt + timedelta(minutes=10)
        pred_time_formatted = pred_time.strftime("%Y-%m-%d %H:%M")
    except:
        pred_time_formatted = "10분 후"
    
    return {
        'current_time': current_time_formatted,
        'pred_time': pred_time_formatted,
        'prediction': prediction,
        'level': level,
        'normal_prob': normal_prob,
        'warning_prob': warning_prob,
        'danger_prob': danger_prob,
        'emoji': emoji,
        'current_value': int(current_totalcnt),
        'seq_max': int(seq_max)
    }

# 메인 실행
if __name__ == "__main__":
    result = predict_next_10min(DATA_FILE, MODEL_DIR)
    
    if result:
        print("="*80)
        print("🔥 ExtraTrees 최종 예측")
        print("="*80)
        print(f"현재시간: {result['current_time']}")
        print(f"예측시간: {result['pred_time']} (10분 후)")
        print("-"*50)
        print(f"예측값: {result['prediction']}")
        print(f"🟢정상확률: {result['normal_prob']}%")
        print(f"🟡주의확률: {result['warning_prob']}%")
        print(f"🔴심각확률: {result['danger_prob']}%")
        
        # 결과 메시지
        if result['level'] == "정상":
            message = "정상 - 안정적"
        elif result['level'] == "주의":
            message = "주의 - 모니터링 필요"
        else:
            message = "심각 - 물류량 급증!"
        
        print(f"결과: {result['emoji']} {message}")
        print("-"*50)
        print(f"[반환값 확인]")
        print(result['prediction'])
        
        # 추가 정보
        print(f"\n[참고 정보]")
        print(f"현재 물류량: {result['current_value']}")
        print(f"280분 중 최대값: {result['seq_max']}")
        
        # CSV 결과 저장
        output_file = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        result_df = pd.DataFrame([{
            '예측시간': result['pred_time'],
            '예측값': result['prediction'],
            '구간': result['level'],
            '정상확률': result['normal_prob'],
            '주의확률': result['warning_prob'],
            '심각확률': result['danger_prob']
        }])
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 결과 저장: {output_file}")
    else:
        print("❌ 예측 실패")
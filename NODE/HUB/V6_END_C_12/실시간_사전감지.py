# -*- coding: utf-8 -*-
"""
실시간 예측 코드 - 사전감지 조건 및 +15 보정 적용
새로운 데이터가 들어올 때마다 10분 후 예측
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta


class RealtimePredictor:
    """실시간 예측기"""
    
    def __init__(self, model_path='xgboost_model_30min_10min_12컬럼.pkl'):
        """모델 로드"""
        
        # 필수 컬럼 정의
        self.FEATURE_COLS = {
            'storage': ['M16A_3F_STORAGE_UTIL'],
            'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
            'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
            'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
            'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
        }
        
        self.TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
        
        # 모델 로드
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ 모델 로드 완료: {model_path}")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
        
        # 과거 30개 데이터 버퍼
        self.data_buffer = []
        
        print("✅ 실시간 예측기 준비 완료!")
        print("\n🚨 사전감지 조건:")
        print("  조건1: 30 시퀀스 MAX < 300")
        print("  조건2: 283 이상 값 존재")
        print("  조건3: 증가율 >= 15")
        print("  보정: 조건 만족 + 예측값 < 300 → +15\n")
    
    
    def predict(self, new_data_row):
        """
        새로운 데이터 1개가 들어올 때 10분 후 예측
        
        Parameters:
        -----------
        new_data_row : dict or pd.Series
            새로운 1분 데이터 (모든 컬럼 포함)
        
        Returns:
        --------
        dict : 예측 결과
        """
        
        # DataFrame으로 변환
        if isinstance(new_data_row, dict):
            new_data_row = pd.Series(new_data_row)
        
        # 버퍼에 추가
        self.data_buffer.append(new_data_row)
        
        # 30개 이상일 때만 예측
        if len(self.data_buffer) < 30:
            return {
                'status': 'waiting',
                'message': f'데이터 수집 중... ({len(self.data_buffer)}/30)',
                'prediction': None
            }
        
        # 최근 30개만 유지
        if len(self.data_buffer) > 30:
            self.data_buffer.pop(0)
        
        # DataFrame 변환
        seq_df = pd.DataFrame(self.data_buffer)
        seq_target = seq_df[self.TARGET_COL].values
        
        # ========================================
        # Feature 생성
        # ========================================
        features = {
            # 타겟 컬럼 특성
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_last_5_mean': np.mean(seq_target[-5:]),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
            'target_last_10_mean': np.mean(seq_target[-10:]),
            'target_first_10_mean': np.mean(seq_target[:10])
        }
        
        # 각 컬럼 그룹별 특성
        for group_name, cols in self.FEATURE_COLS.items():
            for col in cols:
                if col in seq_df.columns:
                    col_seq = seq_df[col].values
                    
                    features[f'{col}_mean'] = np.mean(col_seq)
                    features[f'{col}_std'] = np.std(col_seq)
                    features[f'{col}_max'] = np.max(col_seq)
                    features[f'{col}_min'] = np.min(col_seq)
                    features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
                    features[f'{col}_last_10_mean'] = np.mean(col_seq[-10:])
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
                    features[f'{col}_first_10_mean'] = np.mean(col_seq[:10])
                    features[f'{col}_mid_10_mean'] = np.mean(col_seq[10:20])
                    features[f'{col}_last_value'] = col_seq[-1]
        
        # 유입-유출 차이
        inflow_sum = 0
        outflow_sum = 0
        for col in self.FEATURE_COLS['inflow']:
            if col in seq_df.columns:
                inflow_sum += seq_df[col].iloc[-1]
        for col in self.FEATURE_COLS['outflow']:
            if col in seq_df.columns:
                outflow_sum += seq_df[col].iloc[-1]
        features['net_flow'] = inflow_sum - outflow_sum
        
        # CMD 총합
        cmd_sum = 0
        for col in self.FEATURE_COLS['cmd']:
            if col in seq_df.columns:
                cmd_sum += seq_df[col].iloc[-1]
        features['total_cmd'] = cmd_sum
        
        X_pred = pd.DataFrame([features])
        
        # ========================================
        # 모델 예측
        # ========================================
        prediction = self.model.predict(X_pred)[0]
        
        # ========================================
        # 🚨 사전감지 조건 체크
        # ========================================
        seq_max = np.max(seq_target)
        seq_min = np.min(seq_target)
        increase_rate = seq_target[-1] - seq_target[0]
        
        # 조건 1: MAX < 300
        condition1 = seq_max < 300
        
        # 조건 2: 283 이상 존재
        condition2 = np.any(seq_target >= 283)
        
        # 조건 3: 증가율 >= 15
        condition3 = increase_rate >= 15
        
        # 사전감지 조건
        사전감지 = condition1 and condition2 and condition3
        
        # ========================================
        # 예측값 보정
        # ========================================
        if 사전감지 and prediction < 300:
            최종예측값 = prediction + 15  # +15 보정
            보정여부 = True
        else:
            최종예측값 = prediction
            보정여부 = False
        
        # ========================================
        # 결과 반환
        # ========================================
        result = {
            'status': 'success',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # 시퀀스 정보
            '시퀀스MAX': round(seq_max, 2),
            '시퀀스MIN': round(seq_min, 2),
            '시퀀스평균': round(np.mean(seq_target), 2),
            '시퀀스증가': round(increase_rate, 2),
            
            # 조건 체크
            '조건1_MAX<300': condition1,
            '조건2_283이상': condition2,
            '조건3_증가15이상': condition3,
            '사전감지': 사전감지,
            
            # 예측 결과
            '기본예측값': round(prediction, 2),
            '보정적용': '✅' if 보정여부 else '❌',
            '최종예측값': round(최종예측값, 2),
            
            # 알람
            '알람': '🚨 사전감지!' if 사전감지 else '정상',
            '예측상태': '🔴극단' if 최종예측값 >= 300 else ('🟡주의' if 최종예측값 >= 280 else '🟢정상')
        }
        
        return result
    
    
    def reset(self):
        """버퍼 초기화"""
        self.data_buffer = []
        print("✅ 데이터 버퍼 초기화 완료")


# ========================================
# 사용 예시
# ========================================
if __name__ == '__main__':
    
    print("="*80)
    print("🚀 실시간 예측기 시작")
    print("="*80)
    
    # 1. 예측기 생성
    predictor = RealtimePredictor()
    
    # 2. 테스트 데이터 로드
    print("\n📂 테스트 데이터 로드...")
    df = pd.read_csv('HUB0905101512.CSV', on_bad_lines='skip')
    print(f"✅ 데이터 로드 완료: {len(df)}개 행\n")
    
    # 3. 실시간 시뮬레이션 (첫 50개만 테스트)
    print("="*80)
    print("🔄 실시간 예측 시뮬레이션 (첫 50개)")
    print("="*80)
    
    for i in range(50):
        row = df.iloc[i]
        result = predictor.predict(row)
        
        if result['status'] == 'success':
            print(f"\n[{i+1}번째 데이터]")
            print(f"  시간: {result['timestamp']}")
            print(f"  시퀀스MAX: {result['시퀀스MAX']}, 증가: {result['시퀀스증가']}")
            print(f"  사전감지: {result['사전감지']} (조건1:{result['조건1_MAX<300']}, 조건2:{result['조건2_283이상']}, 조건3:{result['조건3_증가15이상']})")
            print(f"  기본예측: {result['기본예측값']} → 보정:{result['보정적용']} → 최종: {result['최종예측값']}")
            print(f"  {result['알람']} {result['예측상태']}")
        else:
            print(f"[{i+1}번째] {result['message']}")
    
    print("\n" + "="*80)
    print("✅ 실시간 예측 시뮬레이션 완료!")
    print("="*80)
    
    # 4. 실제 사용법 안내
    print("\n" + "="*80)
    print("📖 실제 사용법")
    print("="*80)
    print("""
# 예측기 생성
predictor = RealtimePredictor()

# 새 데이터가 들어올 때마다 호출
while True:
    new_data = get_new_data()  # 새로운 1분 데이터
    result = predictor.predict(new_data)
    
    if result['status'] == 'success':
        print(f"최종예측값: {result['최종예측값']}")
        print(f"알람: {result['알람']}")
        
        # 사전감지 시 알람
        if result['사전감지']:
            send_alarm(result['최종예측값'])
    """)
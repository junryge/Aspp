#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🎯 HUBROOM 점프 감지 시스템 - 최종 실행 코드 (v4.1)
================================================================================
목표:
- 누락된 입력 컬럼을 자동으로 감지하고 생성하는 방어 로직 추가
- Feature Shape Mismatch 오류를 근본적으로 해결
- get_prediction_result() 호출 시 최종 문자열 반환
================================================================================
"""
import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class HubRoomDataProcessor:
    def __init__(self):
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.inflow_cols = ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2', 'M16B_10F_TO_HUB_JOB']
        self.outflow_cols = ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB', 'M16A_3F_TO_3F_MLUD_JOB']
        self.cmd_cols = ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD', 'M16A_2F_TO_HUB_CMD', 'M14A_3F_TO_HUB_CMD', 'M14B_7F_TO_HUB_CMD']
        # 모델 학습에 사용된 21개 필수 원본 컬럼 목록
        self.required_base_cols = self.inflow_cols + self.outflow_cols + self.cmd_cols + \
                                  ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA', 'M16A_3F_STORAGE_UTIL', 'M14_TO_M16_OFS_CUR', 'M16_TO_M14_OFS_CUR', self.target_col]
        self.probability_map = {0: 0.003, 1: 0.15, 2: 0.25, 3: 0.31, 4: 0.43, 5: 0.43, 6: 0.35, 7: 0.42, 8: 0.53, 9: 0.49, 10: 0.42, 11: 0.47, 12: 0.52, 13: 0.60, 14: 0.54, 15: 0.66, 16: 0.62, 17: 0.71, 18: 0.79, 19: 0.83, 20: 0.987, 21: 0.99, 22: 0.99, 23: 0.99, 24: 0.99, 25: 0.99, 26: 0.99, 27: 0.99, 28: 0.99, 29: 0.99, 30: 0.99}

    def load_and_merge_data(self, data_path):
        print("\n[단계 1/3] 데이터 로드 및 정합성 검사...")
        df = pd.read_csv(data_path)
        
        # [강화된 방어 코드] 필수 21개 컬럼이 없으면 자동으로 생성
        for col in self.required_base_cols:
            if col not in df.columns:
                print(f"     ⚠️ 경고: 필수 컬럼 '{col}'이 없어 기본값 0으로 자동 생성합니다.")
                df[col] = 0
        
        df['datetime'] = pd.to_datetime(df[df.columns[0]], format='%Y%m%d%H%M')
        bridge_path = data_path.replace('.csv', '_BRIDGE.csv')
        if os.path.exists(bridge_path):
            bridge_df = pd.read_csv(bridge_path)
            if 'IDC_VAL' in bridge_df.columns:
                bridge_df['BRIDGE_TIME'] = bridge_df['IDC_VAL']; bridge_df['datetime'] = pd.to_datetime(bridge_df['CRT_TM']).dt.floor('min')
                df = pd.merge(df, bridge_df[['datetime', 'BRIDGE_TIME']], on='datetime', how='left')
        if 'BRIDGE_TIME' not in df.columns: df['BRIDGE_TIME'] = 3.5
        df['BRIDGE_TIME'] = df['BRIDGE_TIME'].fillna(3.5)
        return df

    def create_all_features(self, df):
        print("[단계 2/3] 전체 특징 생성 (평가 코드와 동일)...")
        # 평가 코드의 모든 특징 생성 로직을 그대로 사용
        df['flow_balance'] = df[self.inflow_cols].sum(axis=1) - df[self.outflow_cols].sum(axis=1); df['flow_ratio'] = df[self.inflow_cols].sum(axis=1) / (df[self.outflow_cols].sum(axis=1) + 1)
        df['trend_20min'] = df[self.target_col].diff(20); df['trend_10min'] = df[self.target_col].diff(10); df['acceleration'] = df['trend_10min'] - df['trend_10min'].shift(10)
        df['consecutive_250+'] = (df[self.target_col] > 250).rolling(10).sum(); df['consecutive_270+'] = (df[self.target_col] > 270).rolling(10).sum()
        df['cmd_sync_count'] = (df[self.cmd_cols] > 235).sum(axis=1); df['cmd_max'] = df[self.cmd_cols].max(axis=1)
        df['bridge_diff'] = df['BRIDGE_TIME'].diff(5); df['bridge_high'] = (df['BRIDGE_TIME'] > 4.0).astype(int); df['storage_x_bridge'] = df['M16A_3F_STORAGE_UTIL'] * df['BRIDGE_TIME']
        consecutive_300_probs = [self.probability_map.get(sum(1 for v in df[self.target_col].iloc[i-30:i] if v >= 300), 0.5) if i >= 30 else 0.003 for i in range(len(df))]
        df['consecutive_300_prob'] = consecutive_300_probs
        df['change_rate_10min'] = ((df[self.target_col] - df[self.target_col].shift(10)) / (df[self.target_col].shift(10) + 1)) * 100
        df['change_rate_20min'] = ((df[self.target_col] - df[self.target_col].shift(20)) / (df[self.target_col].shift(20) + 1)) * 100
        df['change_rate_30min'] = ((df[self.target_col] - df[self.target_col].shift(30)) / (df[self.target_col].shift(30) + 1)) * 100
        df['volatility_10min'] = df[self.target_col].rolling(10).std(); df['volatility_20min'] = df[self.target_col].rolling(20).std(); df['volatility_30min'] = df[self.target_col].rolling(30).std()
        df['distance_to_300'] = 300 - df[self.target_col]; df['near_extreme'] = (df[self.target_col] > 280).astype(int)
        df['recent_5min_mean'] = df[self.target_col].rolling(5).mean(); df['recent_5min_max'] = df[self.target_col].rolling(5).max(); df['recent_10min_mean'] = df[self.target_col].rolling(10).mean()
        df['in_jump_zone'] = ((df[self.target_col] >= 275) & (df[self.target_col] <= 279)).astype(int)
        df['range_class'] = np.select([df[self.target_col] < 150, df[self.target_col] < 300], [0, 1], default=2)
        df['is_jump'] = ((df[self.target_col].rolling(30).max().shift(10) < 280) & (df[self.target_col] >= 300)).astype(int)
        df['change_20min'] = df[self.target_col] - df[self.target_col].shift(20)
        df['trend_pattern'] = np.select([df['change_20min'] < -20, df['change_20min'] < 20, df['change_20min'] < 50], [0, 1, 2], default=3)
        df.fillna(method='ffill', inplace=True); df.fillna(0, inplace=True)
        return df

class RealTimePredictionRunner:
    def __init__(self, data_path='data/HUBROOM_PIVOT_DATA.CSV', models_dir='./checkpoints_jump80/models'):
        print("🚀 HUBROOM 예측 시스템 초기화...")
        if not os.path.exists(data_path): raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        if not os.path.exists(models_dir): raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {models_dir}")
        self.data_path = data_path
        self.processor = HubRoomDataProcessor()
        self.system = {}
        self._load_models(models_dir)

    def _load_models(self, models_dir):
        try:
            for name in ['jump', 'range', 'trend', 'value']: self.system[name] = joblib.load(os.path.join(models_dir, f'model_{name}.pkl'))
            print("✅ 모델 4종 로드 완료.")
        except Exception as e: raise RuntimeError(f"모델 로드에 실패했습니다: {e}")

    @staticmethod
    def _prepare_features(X_seq):
        return np.hstack([X_seq[:, -1, :], np.mean(X_seq, axis=1), np.std(X_seq, axis=1), np.max(X_seq, axis=1), np.min(X_seq, axis=1), X_seq[:, -1, :] - X_seq[:, 0, :]])
    
    @staticmethod
    def _apply_boosts(X_seq, value_pred):
        seq_max = np.max(X_seq[0, :, 0]); recent_5min_mean = np.mean(X_seq[0, -5:, 0])
        if (value_pred[0] >= 290 and seq_max < 280) or (recent_5min_mean >= 275 and seq_max <= 279): return 1
        return 0

    @staticmethod
    def _determine_status(jump_final, jump_proba, value_pred, trend_pred):
        if jump_final == 1: level, percentage = "심각", jump_proba * 100
        else:
            caution_score = jump_proba * 100
            if value_pred > 280: caution_score += 30
            elif value_pred > 260: caution_score += 15
            if trend_pred == 3: caution_score += 25
            elif trend_pred == 2: caution_score += 10
            if caution_score >= 50: level, percentage = "주의", min(caution_score, 95.0)
            else: level, percentage = "안정", 100 - caution_score
        return level, percentage

    def get_prediction_result(self) -> str:
        df_featured = self.processor.create_all_features(self.processor.load_and_merge_data(self.data_path))
        seq_len = 30
        if len(df_featured) < seq_len: return f"오류: 데이터 부족 (필요: {seq_len}개)"
        
        feature_cols = [col for col in df_featured.columns if col not in ['datetime', 'range_class', 'is_jump', 'trend_pattern']]
        X_seq = np.expand_dims(df_featured.tail(seq_len)[feature_cols].values, axis=0)
        
        print("[단계 3/3] 모델 예측 및 최종 결과 생성...")
        X_features = self._prepare_features(X_seq)
        
        jump_proba = self.system['jump'].predict_proba(X_features)[:, 1]
        value_pred = self.system['value'].predict(X_features)
        trend_pred = self.system['trend'].predict(X_features)
        
        jump_final = self._apply_boosts(X_seq, value_pred)
        adjusted_value = value_pred[0] + 40 if jump_final == 1 else value_pred[0]
        risk_korean, risk_percentage = self._determine_status(jump_final, jump_proba[0], adjusted_value, trend_pred[0])
        
        return f"{adjusted_value:.0f},{risk_korean}({risk_percentage:.0f}%)"

if __name__ == "__main__":
    try:
        runner = RealTimePredictionRunner(data_path='data/HUBROOM_PIVOT_DATA.CSV')
        result_string = runner.get_prediction_result()
        print("\n" + "="*50)
        print(f"  ✅ 최종 예측 결과 (문자열): {result_string}")
        print("="*50)
    except Exception as e:
        print(f"\n❌ 치명적인 오류 발생: {e}")
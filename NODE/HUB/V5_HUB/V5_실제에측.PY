#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🎯 HUBROOM 점프 감지 시스템 - 실제 예측 코드 (v3.0 - 문자열 반환)
================================================================================
목표:
- 예측 시스템을 클래스(RealTimePredictionRunner)로 패키징
- get_prediction_result() 호출 시 "285,주의(65%)" 형태의 문자열을 반환
================================================================================
"""
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 데이터 처리 및 모델 시스템 클래스는 내부적으로 사용되므로 그대로 둡니다.
class HubRoomDataProcessor:
    def __init__(self):
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.inflow_cols = ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2', 'M16B_10F_TO_HUB_JOB']
        self.outflow_cols = ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB', 'M16A_3F_TO_3F_MLUD_JOB']
        self.cmd_cols = ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD', 'M16A_2F_TO_HUB_CMD', 'M14A_3F_TO_HUB_CMD', 'M14B_7F_TO_HUB_CMD']
        self.probability_map = {0: 0.003, 1: 0.15, 2: 0.25, 3: 0.31, 4: 0.43, 5: 0.43, 6: 0.35, 7: 0.42, 8: 0.53, 9: 0.49, 10: 0.42, 11: 0.47, 12: 0.52, 13: 0.60, 14: 0.54, 15: 0.66, 16: 0.62, 17: 0.71, 18: 0.79, 19: 0.83, 20: 0.987, 21: 0.99, 22: 0.99, 23: 0.99, 24: 0.99, 25: 0.99, 26: 0.99, 27: 0.99, 28: 0.99, 29: 0.99, 30: 0.99}

    def load_and_merge_data(self, data_path):
        df = pd.read_csv(data_path)
        df['datetime'] = pd.to_datetime(df[df.columns[0]], format='%Y%m%d%H%M')
        bridge_path = data_path.replace('.csv', '_BRIDGE.csv')
        if os.path.exists(bridge_path):
            bridge_df = pd.read_csv(bridge_path)
            if 'IDC_VAL' in bridge_df.columns:
                bridge_df['BRIDGE_TIME'] = bridge_df['IDC_VAL']
                bridge_df['datetime'] = pd.to_datetime(bridge_df['CRT_TM']).dt.floor('min')
                df = pd.merge(df, bridge_df[['datetime', 'BRIDGE_TIME']], on='datetime', how='left')
                df['BRIDGE_TIME'] = df['BRIDGE_TIME'].interpolate(method='linear', limit_direction='both')
        if 'BRIDGE_TIME' not in df.columns: df['BRIDGE_TIME'] = 3.5
        df['BRIDGE_TIME'] = df['BRIDGE_TIME'].fillna(3.5)
        return df

    def create_all_features(self, df):
        df['flow_balance'] = df[self.inflow_cols].sum(axis=1) - df[self.outflow_cols].sum(axis=1)
        df['acceleration'] = (df[self.target_col].diff(10)) - (df[self.target_col].diff(10)).shift(10)
        df['trend_20min'] = df[self.target_col].diff(20)
        df['consecutive_250+'] = (df[self.target_col] > 250).rolling(10).sum()
        df['cmd_sync_count'] = (df[self.cmd_cols] > 235).sum(axis=1)
        consecutive_300_probs = []
        for i in range(len(df)):
            if i < 30: prob = 0.003
            else: count = sum(1 for v in df[self.target_col].iloc[i-30:i] if v >= 300); prob = self.probability_map.get(count, 0.5)
            consecutive_300_probs.append(prob)
        df['consecutive_300_prob'] = consecutive_300_probs
        df['recent_5min_max'] = df[self.target_col].rolling(5).max()
        df['in_jump_zone'] = ((df[self.target_col] >= 275) & (df[self.target_col] <= 279)).astype(int)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(0)
        return df

class JumpDetectionSystem:
    def __init__(self): self.models = {}; self.feature_indices = {}
    def prepare_features(self, X_seq): return np.hstack([X_seq[:, -1, :], np.mean(X_seq, axis=1), np.std(X_seq, axis=1), np.max(X_seq, axis=1), np.min(X_seq, axis=1), X_seq[:, -1, :] - X_seq[:, 0, :]])
    def get_expanded_feature_indices(self, df):
        feature_cols = [c for c in df.columns if c not in ['datetime']]
        n = len(feature_cols)
        base = {'storage_util': feature_cols.index('M16A_3F_STORAGE_UTIL'), 'bridge_time': feature_cols.index('BRIDGE_TIME'), 'flow_balance': feature_cols.index('flow_balance'), 'consecutive_250': feature_cols.index('consecutive_250+'), 'cmd_sync': feature_cols.index('cmd_sync_count'), 'trend_20min': feature_cols.index('trend_20min'), 'acceleration': feature_cols.index('acceleration'), 'prob_extreme': feature_cols.index('consecutive_300_prob'), 'in_jump_zone': feature_cols.index('in_jump_zone'), 'recent_5min_max': feature_cols.index('recent_5min_max')}
        for k, v in base.items(): self.feature_indices[f'{k}_last']=v; self.feature_indices[f'{k}_mean']=v+n; self.feature_indices[f'{k}_std']=v+2*n; self.feature_indices[f'{k}_max']=v+3*n; self.feature_indices[f'{k}_min']=v+4*n; self.feature_indices[f'{k}_trend']=v+5*n
    def apply_boosts(self, X_seq, X_features, predictions, prob_scores, value_pred):
        boosted = predictions.copy(); idx = self.feature_indices
        cond1 = (X_features[:, idx['storage_util_last']] > 15) | (X_features[:, idx['storage_util_max']] > 20); boosted[cond1] = 1
        cond2 = (X_features[:, idx['bridge_time_last']] > 3.8) | (X_features[:, idx['bridge_time_max']] > 4.0); boosted[cond2] = 1
        complex_conds = np.sum([X_features[:, idx['flow_balance_mean']] > 30, X_features[:, idx['acceleration_last']] > 10, X_features[:, idx['consecutive_250_max']] >= 5, X_features[:, idx['trend_20min_last']] > 20], axis=0) >= 2; boosted[complex_conds] = 1
        if prob_scores is not None: boosted[prob_scores > 0.1] = 1
        boosted[X_features[:, idx['in_jump_zone_last']] > 0] = 1; boosted[X_features[:, idx['recent_5min_max_last']] >= 275] = 1
        seq_max = np.max(X_seq[0, :, 0])
        if value_pred[0] >= 290 and seq_max < 280: boosted[0] = 1
        if np.mean(X_seq[0, -5:, 0]) >= 275 and seq_max <= 279: boosted[0] = 1
        return boosted

# ==============================================================================
# ✨ 최종 예측 실행 클래스
# ==============================================================================
class RealTimePredictionRunner:
    """
    모델 로드부터 예측, 최종 결과 문자열 반환까지 모든 과정을 처리하는 클래스.
    """
    def __init__(self, data_path='data/HUBROOM_PIVOT_DATA.CSV', models_dir='./checkpoints_jump80/models'):
        print("🚀 HUBROOM 예측 시스템 초기화...")
        if not os.path.exists(data_path): raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        if not os.path.exists(models_dir): raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {models_dir}")
        
        self.data_path = data_path
        self.processor = HubRoomDataProcessor()
        self.system = JumpDetectionSystem()
        
        # 초기화 시 모델을 한 번만 로드하여 효율성 증대
        self._load_models(models_dir)

    def _load_models(self, models_dir):
        """내부적으로 모델을 로드하는 메서드"""
        try:
            model_names = ['jump', 'range', 'trend', 'value']
            for name in model_names:
                self.system.models[name] = joblib.load(os.path.join(models_dir, f'model_{name}.pkl'))
            print("✅ 모델 4종 로드 완료.")
        except Exception as e:
            raise RuntimeError(f"모델 로드에 실패했습니다: {e}")

    @staticmethod
    def _determine_status(jump_final, jump_proba, value_pred, trend_pred):
        """예측 결과를 바탕으로 최종 상태와 대표 확률을 결정"""
        if jump_final == 1:
            level = "심각"
            percentage = jump_proba * 100
        else:
            caution_score = jump_proba * 100
            if value_pred > 280: caution_score += 30
            elif value_pred > 260: caution_score += 15
            if trend_pred == 3: caution_score += 25
            elif trend_pred == 2: caution_score += 10
            
            if caution_score >= 50:
                level = "주의"
                percentage = min(caution_score, 95.0)
            else:
                level = "안정"
                percentage = 100 - caution_score
        return level, percentage

    def get_prediction_result(self) -> str:
        """
        최신 데이터를 읽어 10분 후를 예측하고, 결과를 포맷에 맞는 문자열로 반환합니다.
        
        :return: "예측값,상태(확률%)" 형식의 문자열. 예: "285,주의(65%)"
        """
        # 1. 데이터 로드 및 특징 생성
        df_raw = self.processor.load_and_merge_data(self.data_path)
        df_featured = self.processor.create_all_features(df_raw)

        # 2. 예측에 사용할 최신 시퀀스 추출
        seq_len = 30
        if len(df_featured) < seq_len:
            return f"오류: 데이터 부족 (필요: {seq_len}개)"
        
        latest_data = df_featured.tail(seq_len)
        feature_cols = [c for c in df_featured.columns if c not in ['datetime']]
        X_seq = np.expand_dims(latest_data[feature_cols].values, axis=0)

        # 3. 모델 예측 수행
        self.system.get_expanded_feature_indices(df_featured)
        X_features = self.system.prepare_features(X_seq)
        
        jump_proba = self.system.models['jump'].predict_proba(X_features)[:, 1]
        jump_pred = (jump_proba > 0.5).astype(int)
        value_pred = self.system.models['value'].predict(X_features)
        
        # 4. 부스팅 규칙 적용
        jump_final = self.system.apply_boosts(X_seq, X_features, jump_pred, jump_proba, value_pred)
        
        # 5. 최종 결과 계산 및 포맷팅
        adjusted_value = value_pred[0] + 40 if jump_final[0] == 1 else value_pred[0]
        trend_pred = self.system.models['trend'].predict(X_features)
        
        risk_korean, risk_percentage = self._determine_status(
            jump_final=jump_final[0],
            jump_proba=jump_proba[0],
            value_pred=adjusted_value,
            trend_pred=trend_pred[0]
        )
        
        return f"{adjusted_value:.0f},{risk_korean}({risk_percentage:.0f}%)"

# ==============================================================================
# 🚀 실행 예시
# ==============================================================================
if __name__ == "__main__":
    try:
        # 1. 예측 실행기(Runner) 객체 생성 (이때 모델들이 로드됩니다)
        runner = RealTimePredictionRunner()
        
        # 2. 예측 결과 요청
        print("\n[요청] 최신 데이터 기반으로 10분 후 예측 수행...")
        result_string = runner.get_prediction_result()
        
        # 3. 반환된 문자열 결과 출력
        print("\n" + "="*50)
        print(f"  ✅ 최종 예측 결과 (문자열): {result_string}")
        print("="*50)

    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"\n❌ 치명적인 오류 발생: {e}")
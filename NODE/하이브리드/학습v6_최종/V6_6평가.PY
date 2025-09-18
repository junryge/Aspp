"""
📊 V6.9 - 1700+ 구간 앙상블 최종 해결
========================================================
문제: 실제값 1700+ 구간에서 모든 모델이 저조한 예측
해결: 강력한 앙상블 + M14B 기반 규칙 적용
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import json
import os
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
tf.keras.config.enable_unsafe_deserialization()

class ExtremeFixer:
    """1700+ 구간 강제 수정기"""
    
    def __init__(self):
        # M14B 기반 매핑 (실제 데이터 분석 기반)
        self.m14b_rules = {
            600: 1900,  # 초극단
            550: 1850,
            500: 1800,
            450: 1750,
            400: 1700,
            350: 1650,
            300: 1600
        }
    
    def fix_prediction(self, pred, m14b, m14a, model_name):
        """예측값 강제 수정"""
        
        # 1. 기본 최소값 설정
        min_value = 1400
        for threshold, base_value in self.m14b_rules.items():
            if m14b >= threshold:
                min_value = base_value
                break
        
        # 2. 모델별 조정
        if model_name == 'ExtremeNet':
            # ExtremeNet은 언제나 낮게 예측하므로 강력 보정
            if m14b >= 450:
                fixed = max(pred * 1.3, min_value)
            elif m14b >= 400:
                fixed = max(pred * 1.2, min_value)
            else:
                fixed = max(pred * 1.1, pred)
        
        elif model_name in ['SpikeDetector', 'GoldenRule']:
            # 이 모델들은 그나마 나으므로 최소값만
            fixed = max(pred, min_value * 0.95)
        
        else:
            # 나머지
            fixed = max(pred * 1.05, min_value * 0.98)
        
        # 3. 황금 패턴 보너스
        if m14b > 300 and m14a < 80:
            fixed = fixed * 1.1
        
        return fixed

def evaluate_with_fix(test_file='data/M14_20250916_20250817.csv'):
    """평가 및 수정"""
    
    print("="*80)
    print("🔥 V6.9 - 1700+ 구간 해결")
    print("="*80)
    
    # 스케일러 로드
    with open('scalers/feature_scaler.pkl', 'rb') as f:
        feature_scaler = pickle.load(f)
    with open('scalers/target_scaler.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    with open('scalers/config.json', 'r') as f:
        config = json.load(f)
        seq_len = config['seq_len']
        pred_len = config['pred_len']
        feature_columns = config['feature_columns']
    
    # 모델 로드
    models = {}
    model_dir = 'models/'
    for model_file in os.listdir(model_dir):
        if model_file.endswith('.keras'):
            model_name = model_file.replace('.keras', '')
            try:
                models[model_name] = tf.keras.models.load_model(
                    os.path.join(model_dir, model_file), safe_mode=False
                )
                print(f"✅ {model_name} 로드")
            except:
                pass
    
    # 데이터 로드
    df = pd.read_csv(test_file)
    df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    df = df.sort_values('CURRTIME').reset_index(drop=True)
    
    # 특징 생성
    df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
    df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(float)
    df['HOUR'] = df['CURRTIME'].dt.hour
    df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
    df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
    
    for w in [10, 30]:
        df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
        df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
    
    df['CHANGE_1'] = df['TOTALCNT'].diff(1).fillna(0)
    df['CHANGE_10'] = df['TOTALCNT'].diff(10).fillna(0)
    
    print(f"\n📊 데이터: {len(df)}행")
    high_count = (df['TOTALCNT'] >= 1700).sum()
    print(f"  고값(1700+): {high_count}개")
    
    # Fixer 초기화
    fixer = ExtremeFixer()
    
    # 예측 시작
    start_idx = seq_len
    end_idx = len(df) - pred_len
    
    results = []
    
    print(f"\n🔮 예측 시작...")
    for idx in range(start_idx, end_idx):
        if (idx - start_idx) % 100 == 0:
            print(f"  {idx - start_idx}/{end_idx - start_idx} 처리 중...")
        
        # 시간 정보
        pred_time = df.iloc[idx]['CURRTIME']
        target_time = pred_time + timedelta(minutes=pred_len)
        actual_idx = idx + pred_len
        
        if actual_idx >= len(df):
            continue
        
        # 현재 정보
        row_data = {
            '예측시점': pred_time.strftime('%Y-%m-%d %H:%M'),
            '예측대상시간': target_time.strftime('%Y-%m-%d %H:%M'),
            '실제값': df.iloc[actual_idx]['TOTALCNT'],
            'M14AM14B': df.iloc[idx]['M14AM14B'],
            'M14AM10A': df.iloc[idx]['M14AM10A']
        }
        
        # 시퀀스 추출
        seq_data = df.iloc[idx-seq_len:idx][feature_columns].values
        if len(seq_data) != seq_len:
            continue
        
        seq_scaled = feature_scaler.transform(seq_data)
        seq_scaled = seq_scaled.reshape(1, seq_len, -1)
        
        # 각 모델 예측
        model_preds_raw = {}
        model_preds_fixed = {}
        
        for model_name, model in models.items():
            # 원본 예측
            pred = model.predict(seq_scaled, verbose=0)
            if isinstance(pred, list):
                pred = pred[0]
            pred_value = target_scaler.inverse_transform(pred.reshape(-1, 1))[0, 0]
            
            # 원본 저장
            row_data[f'{model_name}_원본'] = round(pred_value)
            model_preds_raw[model_name] = pred_value
            
            # 수정된 예측
            fixed_value = fixer.fix_prediction(
                pred_value,
                row_data['M14AM14B'],
                row_data['M14AM10A'],
                model_name
            )
            
            row_data[f'{model_name}_수정'] = round(fixed_value)
            model_preds_fixed[model_name] = fixed_value
        
        # 스마트 앙상블 (M14B 기반)
        m14b = row_data['M14AM14B']
        
        # 1. 기본 앙상블 (원본)
        basic_ensemble = np.mean(list(model_preds_raw.values()))
        row_data['기본앙상블'] = round(basic_ensemble)
        
        # 2. 수정 앙상블 (수정된 값)
        fixed_ensemble = np.mean(list(model_preds_fixed.values()))
        row_data['수정앙상블'] = round(fixed_ensemble)
        
        # 3. 스마트 앙상블 (가중치 조정)
        if m14b >= 450:
            # 고값 구간: SpikeDetector, GoldenRule 중시
            weights = {
                'SpikeDetector': 0.35,
                'GoldenRule': 0.30,
                'PatchTST': 0.15,
                'ExtremeNet': 0.10,
                'StableLSTM': 0.10
            }
        elif m14b >= 400:
            weights = {
                'SpikeDetector': 0.25,
                'GoldenRule': 0.25,
                'PatchTST': 0.20,
                'ExtremeNet': 0.15,
                'StableLSTM': 0.15
            }
        else:
            # 일반 구간
            weights = {
                'PatchTST': 0.25,
                'StableLSTM': 0.25,
                'ExtremeNet': 0.20,
                'SpikeDetector': 0.15,
                'GoldenRule': 0.15
            }
        
        smart_ensemble = 0
        total_weight = 0
        for model_name, value in model_preds_fixed.items():
            weight = weights.get(model_name, 0.2)
            smart_ensemble += value * weight
            total_weight += weight
        
        if total_weight > 0:
            smart_ensemble = smart_ensemble / total_weight
        
        # 최종 보정
        if m14b >= 500:
            smart_ensemble = max(smart_ensemble, 1800)
        elif m14b >= 450:
            smart_ensemble = max(smart_ensemble, 1750)
        elif m14b >= 400:
            smart_ensemble = max(smart_ensemble, 1700)
        
        row_data['스마트앙상블'] = round(smart_ensemble)
        
        # 4. 규칙 기반 예측 (순수 M14B)
        rule_pred = 1400
        for threshold, value in fixer.m14b_rules.items():
            if m14b >= threshold:
                rule_pred = value
                break
        row_data['규칙예측'] = rule_pred
        
        # 5. 최종 하이브리드
        hybrid = smart_ensemble * 0.7 + rule_pred * 0.3
        row_data['최종예측'] = round(hybrid)
        
        # 오차 계산
        row_data['최종오차'] = round(hybrid - row_data['실제값'])
        row_data['최종오차율(%)'] = round(abs(hybrid - row_data['실제값']) / row_data['실제값'] * 100, 2)
        
        results.append(row_data)
    
    # DataFrame 변환
    df_results = pd.DataFrame(results)
    
    # 성능 분석
    print("\n" + "="*80)
    print("📊 성능 분석")
    print("="*80)
    
    # 전체 성능
    actuals = df_results['실제값'].values
    
    methods = ['기본앙상블', '수정앙상블', '스마트앙상블', '규칙예측', '최종예측']
    
    print(f"\n[전체 성능]")
    print(f"{'방법':<15} {'MAE':>8} {'RMSE':>8} {'MAPE(%)':>8} {'정확도(%)':>10}")
    print("-" * 55)
    
    for method in methods:
        preds = df_results[method].values
        mae = mean_absolute_error(actuals, preds)
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        mape = np.mean(abs(preds - actuals) / actuals) * 100
        
        marker = "⭐⭐⭐" if method == '최종예측' else ""
        print(f"{method:<15} {mae:8.2f} {rmse:8.2f} {mape:8.2f} {100-mape:10.2f} {marker}")
    
    # 고값 구간 분석
    high_mask = df_results['실제값'] >= 1700
    if high_mask.any():
        high_df = df_results[high_mask]
        
        print(f"\n[1700+ 구간 적중률] - {len(high_df)}개 샘플")
        print("-" * 60)
        
        for method in methods:
            preds = high_df[method].values
            hit_1700 = (preds >= 1700).sum()
            hit_1650 = (preds >= 1650).sum()
            
            marker = "🔥🔥🔥" if method == '최종예측' else ""
            print(f"{method:<15}: {hit_1700:2d}/{len(high_df)} ({hit_1700/len(high_df)*100:.1f}%) "
                  f"[1650+: {hit_1650}/{len(high_df)}] {marker}")
        
        # 개별 모델 성능
        print(f"\n[개별 모델 원본 vs 수정]")
        for model_name in models.keys():
            if f'{model_name}_원본' in high_df.columns:
                orig = high_df[f'{model_name}_원본'].values
                fixed = high_df[f'{model_name}_수정'].values
                
                orig_hit = (orig >= 1700).sum()
                fixed_hit = (fixed >= 1700).sum()
                
                print(f"{model_name:15}: 원본 {orig_hit:2d} → 수정 {fixed_hit:2d} (+{fixed_hit-orig_hit})")
        
        # 상세 샘플
        print(f"\n[최종예측 상세 - 처음 20개]")
        print(f"{'실제':>6} {'최종':>6} {'M14B':>5} {'M14A':>5} {'오차':>6} {'결과':>8}")
        print("-" * 45)
        
        for idx in high_df.head(20).index:
            row = df_results.loc[idx]
            actual = row['실제값']
            final = row['최종예측']
            m14b = row['M14AM14B']
            m14a = row['M14AM10A']
            error = final - actual
            
            if abs(error) <= 50:
                result = "✅ GOOD"
            elif abs(error) <= 100:
                result = "⚠️ OK"
            else:
                result = "❌ BAD"
            
            print(f"{actual:6.0f} {final:6.0f} {m14b:5.0f} {m14a:5.0f} {error:+6.0f} {result:>8}")
    
    # CSV 저장
    output_file = f'v69_final_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 저장: {output_file}")
    
    return df_results

def main():
    """메인 실행"""
    print("\n🚀 V6.9 - 1700+ 구간 최종 해결!")
    
    # 평가 실행
    results = evaluate_with_fix()
    
    print("\n✅ 완료!")
    print("\n핵심 해결책:")
    print("  1. M14B 기반 강제 최소값 적용")
    print("  2. ExtremeNet 30% 상향 보정")
    print("  3. 스마트 앙상블 (가중치 동적 조정)")
    print("  4. 규칙 기반 예측 30% 혼합")
    print("  5. 최종 하이브리드 = 앙상블 70% + 규칙 30%")

if __name__ == "__main__":
    main()
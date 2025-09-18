"""
📊 V6.9 극단값 예측 개선 - 성능 및 구조 최적화 버전
============================================================
- 주요 개선 사항:
  1. Vectorization: 순차 처리(for-loop)를 병렬 배치 처리로 변경하여 예측 속도 대폭 향상
  2. Modularization: 기능별 함수 분리로 코드 가독성 및 유지보수성 증대
  3. Configuration: 부스팅 및 앙상블 규칙을 설정으로 분리하여 유연성 확보
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import json
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

# 초기 설정
warnings.filterwarnings('ignore')
tf.keras.config.enable_unsafe_deserialization()


class OptimizedModelEvaluator:
    """
    최적화된 모델 평가기 클래스.
    벡터화 연산을 통해 평가 속도를 극대화하고, 코드 구조를 개선하여 유지보수성을 높였습니다.
    """
    
    def __init__(self, scaler_path: str = 'scalers/', model_dir: str = 'models/'):
        """평가기 초기화 및 리소스 로드"""
        print("="*80)
        print("🔥 V6.9 극단값 예측 평가 (성능/구조 최적화 버전)")
        print("="*80)
        
        self._load_scalers_and_config(scaler_path)
        self.models = self._load_all_models(model_dir)
        
        # 부스팅 및 앙상블 전략 설정
        self.boosting_config = self._get_boosting_config()
        self.ensemble_weights = self._get_ensemble_weights()

    def _load_scalers_and_config(self, scaler_path: str):
        """스케일러와 설정 파일 로드"""
        print("🔄 스케일러 및 설정 로딩...")
        with open(os.path.join(scaler_path, 'feature_scaler.pkl'), 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(os.path.join(scaler_path, 'target_scaler.pkl'), 'rb') as f:
            self.target_scaler = pickle.load(f)
        with open(os.path.join(scaler_path, 'config.json'), 'r') as f:
            config = json.load(f)
            self.seq_len = config['seq_len']
            self.pred_len = config['pred_len']
            self.feature_columns = config['feature_columns']
        
        print(f"  ✅ 설정: {self.seq_len}분 시퀀스 → {self.pred_len}분 후 예측")

    def _load_all_models(self, model_dir: str) -> Dict[str, tf.keras.Model]:
        """디렉토리 내 모든 Keras 모델 로드"""
        print(f"📁 모델 로딩 (경로: {model_dir})")
        models = {}
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
        
        for model_file in model_files:
            model_name = model_file.replace('.keras', '')
            model_path = os.path.join(model_dir, model_file)
            try:
                models[model_name] = tf.keras.models.load_model(model_path, safe_mode=False)
                print(f"  ✅ {model_name} 로드 완료")
            except Exception as e:
                print(f"  ❌ {model_name} 로드 실패: {e}")
        
        print(f"총 {len(models)}개 모델 로드 완료")
        return models

    def _get_boosting_config(self) -> Dict[str, Any]:
        """모델별 극단값 부스팅 규칙 설정"""
        return {
            'default': [
                {'threshold': 500, 'multiply': 1.15, 'min_value': 1750},
                {'threshold': 450, 'multiply': 1.10, 'min_value': 1700},
                {'threshold': 400, 'multiply': 1.05, 'min_value': 1650},
            ],
            'ExtremeNet': [
                {'threshold': 500, 'multiply': 1.30, 'min_value': 1750},
                {'threshold': 450, 'multiply': 1.20, 'min_value': 1700},
                {'threshold': 400, 'multiply': 1.10, 'min_value': 1650},
                {'threshold': 0,   'multiply': 1.05, 'min_value': 0},
            ],
            'SpikeDetector': [
                {'threshold': 500, 'multiply': 1.0, 'min_value': 1750},
                {'threshold': 450, 'multiply': 1.0, 'min_value': 1700},
                {'threshold': 400, 'multiply': 1.0, 'min_value': 1650},
            ],
            'GoldenRule': [ # SpikeDetector와 동일한 규칙 적용
                {'threshold': 500, 'multiply': 1.0, 'min_value': 1750},
                {'threshold': 450, 'multiply': 1.0, 'min_value': 1700},
                {'threshold': 400, 'multiply': 1.0, 'min_value': 1650},
            ],
            'golden_pattern': {'m14b_thr': 300, 'm14a_thr': 80, 'multiply': 1.08}
        }

    def _get_ensemble_weights(self) -> Dict[str, Any]:
        """M14B 값에 따른 앙상블 가중치 설정"""
        return {
            'high': { # M14B > 500
                'weights': {'SpikeDetector': 0.35, 'GoldenRule': 0.30, 'PatchTST': 0.15, 'StableLSTM': 0.10, 'ExtremeNet': 0.10},
                'min_value': 1750
            },
            'medium': { # M14B > 450
                'weights': {'SpikeDetector': 0.30, 'GoldenRule': 0.25, 'PatchTST': 0.20, 'StableLSTM': 0.15, 'ExtremeNet': 0.10},
                'min_value': 1700
            },
            'low': { # M14B > 400
                'weights': {'PatchTST': 0.30, 'StableLSTM': 0.25, 'ExtremeNet': 0.20, 'SpikeDetector': 0.15, 'GoldenRule': 0.10},
                 'min_value': 1650
            },
            'default': { # 그 외
                'weights': {'PatchTST': 0.30, 'StableLSTM': 0.25, 'ExtremeNet': 0.20, 'SpikeDetector': 0.15, 'GoldenRule': 0.10},
                'min_value': 0
            }
        }

    def _load_and_prepare_data(self, filepath: str) -> pd.DataFrame:
        """데이터 로드 및 전처리, 특성 생성"""
        print(f"\n📊 데이터 로딩 및 전처리 (파일: {filepath})")
        df = pd.read_csv(filepath)
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        # 특성 생성 (Vectorized)
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

        print(f"  ✅ 데이터 {len(df)}행 준비 완료")
        high_count = (df['TOTALCNT'] >= 1700).sum()
        print(f"  - 고값(1700 이상) 샘플: {high_count}개 ({high_count/len(df)*100:.1f}%)")
        
        return df

    def _create_sequences(self, df: pd.DataFrame) -> Tuple:
        """데이터프레임으로부터 시퀀스 데이터 생성"""
        print("⏳ 시퀀스 데이터 생성 중...")
        
        # NumPy 배열로 변환하여 속도 향상
        feature_array = df[self.feature_columns].values
        
        # 효율적인 시퀀스 생성을 위한 stride_tricks 사용
        shape = (feature_array.shape[0] - self.seq_len - self.pred_len + 1, self.seq_len, feature_array.shape[1])
        strides = (feature_array.strides[0], feature_array.strides[0], feature_array.strides[1])
        sequences = np.lib.stride_tricks.as_strided(feature_array, shape=shape, strides=strides)

        # 예측 대상이 되는 데이터(y) 및 메타 정보 추출
        indices = np.arange(self.seq_len, len(df) - self.pred_len + 1)
        
        meta_df = df.iloc[indices - 1].copy() # 각 시퀀스의 마지막 행 정보
        meta_df['target_time'] = meta_df['CURRTIME'] + timedelta(minutes=self.pred_len)
        meta_df['actual'] = df['TOTALCNT'].iloc[indices + self.pred_len - 1].values
        
        print(f"  ✅ 총 {len(sequences)}개의 평가 시퀀스 생성 완료")
        return sequences, meta_df.reset_index(drop=True)

    def _predict_in_batches(self, sequences: np.ndarray, batch_size: int = 256) -> Dict[str, np.ndarray]:
        """모든 모델에 대해 배치 단위로 예측 수행"""
        print(f"🔮 모델 예측 시작 (배치 사이즈: {batch_size})")
        
        scaled_sequences = self.feature_scaler.transform(sequences.reshape(-1, sequences.shape[-1]))
        scaled_sequences = scaled_sequences.reshape(sequences.shape)
        
        all_predictions = {}
        for model_name, model in self.models.items():
            print(f"  - {model_name} 예측 중...")
            raw_preds = model.predict(scaled_sequences, batch_size=batch_size, verbose=0)
            if isinstance(raw_preds, list): # Multi-output 모델 처리
                raw_preds = raw_preds[0]
            
            # 역스케일링
            unscaled_preds = self.target_scaler.inverse_transform(raw_preds)
            all_predictions[model_name] = unscaled_preds.flatten()
            
        print("  ✅ 모든 모델 예측 완료")
        return all_predictions

    def _apply_boosting_and_ensemble(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """벡터화 연산으로 부스팅 및 앙상블 적용"""
        print("🛠️ 부스팅 및 앙상블 계산 중...")
        df = predictions_df.copy()
        
        # 1. 모델별 부스팅 적용
        for model_name in self.models.keys():
            preds = df[f'{model_name}_원본'].values
            boosted_preds = preds.copy()
            
            rules = self.boosting_config.get(model_name, self.boosting_config['default'])
            
            for rule in rules:
                mask = (df['M14AM14B'] > rule['threshold'])
                boosted_preds[mask] = np.maximum(preds[mask] * rule['multiply'], rule['min_value'])
            
            # 황금 패턴 부스팅
            gp_rule = self.boosting_config['golden_pattern']
            golden_mask = (df['M14AM14B'] > gp_rule['m14b_thr']) & (df['M14AM10A'] < gp_rule['m14a_thr'])
            boosted_preds[golden_mask] *= gp_rule['multiply']
            
            df[f'{model_name}_예측'] = boosted_preds

        # 2. 극단값 앙상블 계산
        ensemble_preds = np.zeros(len(df))
        
        # M14B 값에 따라 다른 가중치 적용
        conditions = {
            'high': df['M14AM14B'] > 500,
            'medium': (df['M14AM14B'] > 450) & (df['M14AM14B'] <= 500),
            'low': (df['M14AM14B'] > 400) & (df['M14AM14B'] <= 450)
        }
        
        # 기본 가중치 적용 (default)
        weights = self.ensemble_weights['default']['weights']
        weighted_sum = sum(df[f'{name}_예측'] * w for name, w in weights.items() if f'{name}_예측' in df)
        total_weight = sum(weights.values())
        ensemble_preds = weighted_sum / total_weight

        # 조건별 가중치 및 최소값 재계산
        for level, mask in conditions.items():
            config = self.ensemble_weights[level]
            weights = config['weights']
            
            weighted_sum = sum(df.loc[mask, f'{name}_예측'] * w for name, w in weights.items() if f'{name}_예측' in df)
            total_weight = sum(weights.values())
            
            level_ensemble = weighted_sum / total_weight
            level_ensemble = np.maximum(level_ensemble, config['min_value'])
            ensemble_preds[mask] = level_ensemble

        df['극단앙상블_예측'] = ensemble_preds
        print("  ✅ 계산 완료")
        return df

    def _analyze_and_save_results(self, final_df: pd.DataFrame):
        """결과 분석, 출력 및 CSV 저장"""
        print("\n" + "="*80)
        print("📊 최종 성능 분석")
        print("="*80)

        models_to_eval = list(self.models.keys()) + ['극단앙상블']
        actuals = final_df['actual'].values
        
        # 전체 성능
        print(f"\n[전체 성능]")
        print(f"{'모델':<15} {'MAE':>8} {'RMSE':>8} {'MAPE(%)':>8} {'정확도(%)':>10}")
        print("-" * 55)
        
        for model_name in models_to_eval:
            preds = final_df[f'{model_name}_예측'].values
            mae = mean_absolute_error(actuals, preds)
            rmse = np.sqrt(mean_squared_error(actuals, preds))
            mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
            
            if model_name == '극단앙상블':
                print(f"{'🔥 ' + model_name:<15} {mae:8.2f} {rmse:8.2f} {mape:8.2f} {100-mape:10.2f} ⭐")
            else:
                print(f"{model_name:<15} {mae:8.2f} {rmse:8.2f} {mape:8.2f} {100-mape:10.2f}")

        # 고값 구간 분석
        high_mask = final_df['actual'] >= 1700
        if high_mask.any():
            high_df = final_df[high_mask]
            print(f"\n[고값(1700+) 구간 분석] - {len(high_df)}개 샘플")
            print("-" * 75)
            
            for model_name in models_to_eval:
                preds = high_df[f'{model_name}_예측'].values
                hit_count = (preds >= 1700).sum()
                
                if model_name != '극단앙상블':
                    orig_preds = high_df[f'{model_name}_원본'].values
                    orig_hit_count = (orig_preds >= 1700).sum()
                    improvement = hit_count - orig_hit_count
                    print(f"{model_name:15}: 원본 {orig_hit_count:2d} → 부스팅 {hit_count:2d}/{len(high_df)} "
                          f"({hit_count/len(high_df)*100:.1f}%) [개선 +{improvement}]")
                else:
                    print(f"{'🔥 ' + model_name:15}: {hit_count:2d}/{len(high_df)} "
                          f"({hit_count/len(high_df)*100:.1f}%)")

            # 상세 샘플 출력
            print(f"\n[극단앙상블 고값 예측 상세 (상위 15개)]")
            print(f"{'실제':>6} {'예측':>6} {'M14B':>5} {'M14A':>5} {'오차':>6} {'결과':>8}")
            print("-" * 45)
            
            for _, row in high_df.head(15).iterrows():
                actual, pred, m14b, m14a = row['actual'], row['극단앙상블_예측'], row['M14AM14B'], row['M14AM10A']
                error = pred - actual
                result = "✅ HIT" if pred >= 1700 else "⚠️ NEAR" if pred >= 1650 else "❌ MISS"
                print(f"{actual:6.0f} {pred:6.0f} {m14b:5.0f} {m14a:5.0f} {error:+6.0f} {result:>8}")

        # CSV 저장
        output_file = f'v69_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 결과 저장 완료: {output_file}")

    def run_evaluation(self, test_filepath: str):
        """전체 평가 파이프라인 실행"""
        if not self.models:
            print("❌ 로드된 모델이 없어 평가를 진행할 수 없습니다.")
            return

        # 1. 데이터 로드 및 전처리
        df = self._load_and_prepare_data(test_filepath)
        
        # 2. 시퀀스 생성
        sequences, meta_df = self._create_sequences(df)
        if len(sequences) == 0:
            print("❌ 평가할 시퀀스가 부족합니다.")
            return

        # 3. 모델 예측 (배치)
        predictions = self._predict_in_batches(sequences)
        
        # 4. 결과 통합
        # 원본 예측값 저장
        preds_df = pd.DataFrame({f"{name}_원본": preds for name, preds in predictions.items()})
        # 메타 정보와 결합
        results_df = pd.concat([meta_df, preds_df], axis=1)

        # 5. 부스팅 및 앙상블 적용
        final_df = self._apply_boosting_and_ensemble(results_df)

        # 6. 최종 결과 분석 및 저장
        self._analyze_and_save_results(final_df)


def main():
    """메인 실행 함수"""
    print("\n🚀 V6.9 극단값 개선 평가 시작")
    
    # 평가기 생성
    evaluator = OptimizedModelEvaluator(scaler_path='scalers/', model_dir='models/')
    
    # 테스트 파일 경로 탐색
    test_files = [
        'data/20250731_to20250806.csv',
        'data/test_data.csv',
        '/mnt/user-data/uploads/test.csv'
    ]
    
    test_file_path = next((f for f in test_files if os.path.exists(f)), None)
    
    if not test_file_path:
        print("❌ 테스트 데이터를 찾을 수 없습니다. 경로를 확인해주세요.")
        return
    
    # 평가 실행
    evaluator.run_evaluation(test_file_path)
    
    print("\n✅ 모든 평가가 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    main()

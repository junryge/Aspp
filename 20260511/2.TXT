#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML 추론기 — 학습된 XGBoost 모델 로드 + 30분 사전 정체 확률 예측

사용법 (모듈로):
    from ml_predictor import MLPredictor
    ml = MLPredictor('model.json')
    score = ml.predict(features_dict)   # 0.0 ~ 1.0
    level = ml.level(score)             # 'OK' / 'INFO' / 'WARNING' / 'CRITICAL'

CLI 사용:
    python3 ml_predictor.py --model model.json --features features.csv
"""

import argparse
import json
import os
import sys


class MLPredictor:
    """학습된 XGBoost 모델로 30분 사전 확률 예측 (sklearn 불필요 — Booster 직접 사용)"""

    def __init__(self, model_path='model.json', feature_names_path=None):
        import xgboost as xgb
        # Booster 직접 사용 (sklearn 의존성 회피)
        self.booster = xgb.Booster()
        self.booster.load_model(model_path)

        # 피처 순서 로드
        if feature_names_path is None:
            feature_names_path = model_path.replace('.json', '_features.json').replace('.bin', '_features.json')

        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r', encoding='utf-8') as f:
                self.feature_names = json.load(f)
        else:
            # Booster 에서 직접 추출 시도
            self.feature_names = self.booster.feature_names or []

        if not self.feature_names:
            raise ValueError(f'피처명 로드 실패: {feature_names_path}')

        # 호환 alias
        self.model = self.booster

    def _to_dmatrix(self, X):
        import xgboost as xgb
        return xgb.DMatrix(X, feature_names=self.feature_names)

    def predict(self, features_dict):
        """단일 시점 피처 dict → 0~1 확률"""
        import numpy as np
        x = np.array([[features_dict.get(n, 0) or 0 for n in self.feature_names]], dtype=float)
        prob = self.booster.predict(self._to_dmatrix(x))[0]
        return float(prob)

    def predict_batch(self, features_list):
        """여러 시점 피처 리스트 → 확률 리스트"""
        import numpy as np
        X = np.array(
            [[fd.get(n, 0) or 0 for n in self.feature_names] for fd in features_list],
            dtype=float,
        )
        probs = self.booster.predict(self._to_dmatrix(X))
        return [float(p) for p in probs]

    @staticmethod
    def level(prob):
        """확률 → 위험 레벨"""
        if prob >= 0.85:
            return 'CRITICAL'
        if prob >= 0.70:
            return 'WARNING'
        if prob >= 0.30:
            return 'INFO'
        return 'OK'

    @staticmethod
    def level_korean(prob):
        if prob >= 0.85:
            return '위험'
        if prob >= 0.70:
            return '경보'
        if prob >= 0.30:
            return '주의'
        return '정상'


def cli_predict(model_path, features_csv, out_path=None):
    """CLI: 피처 CSV 전체에 확률 부여"""
    import csv
    ml = MLPredictor(model_path)

    rows_out = []
    with open(features_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 숫자형 변환
            features = {}
            for k, v in row.items():
                if k == 'timestamp':
                    continue
                try:
                    features[k] = float(v) if v not in ('', None) else 0
                except (ValueError, TypeError):
                    features[k] = 0

            score = ml.predict(features)
            rows_out.append({
                'timestamp': row.get('timestamp', ''),
                'ml_score': f'{score:.4f}',
                'ml_level': ml.level(score),
                'ml_level_kr': ml.level_korean(score),
            })

    if out_path is None:
        out_path = features_csv.replace('.csv', '_pred.csv')

    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['timestamp', 'ml_score', 'ml_level', 'ml_level_kr'])
        w.writeheader()
        w.writerows(rows_out)

    # 분포 통계
    levels = [r['ml_level'] for r in rows_out]
    from collections import Counter
    print(f'✅ {len(rows_out):,} 행 예측 → {out_path}')
    print(f'   레벨 분포: {dict(Counter(levels))}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='학습된 모델 (model.json)')
    p.add_argument('--features', required=True, help='피처 CSV')
    p.add_argument('--out', help='출력 CSV (생략 시 _pred.csv 자동)')
    args = p.parse_args()

    cli_predict(args.model, args.features, args.out)


if __name__ == '__main__':
    main()

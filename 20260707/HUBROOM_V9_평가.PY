import argparse
import glob
import importlib.machinery
import importlib.util
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ============================================================
# 🎯 V9 평가 — 학습 안 본 6월(OUT_6)로 검증
# ------------------------------------------------------------
# ① 회귀 성능: 10분후 MIN/MAX/AVG 변화량 MAE/RMSE/R²
# ② 급등(+50↑) 방향 예측: 컷별 정밀도/재현율
# ③ (옵션 --alerts) CUSUM 경보 오탐필터 성능:
#     판정 완료된 밀림_경보요약.csv 와 대조 →
#     "예측MAX ≥ 컷" 필터를 걸면 오탐 몇 % 제거 / 정탐 몇 % 보존인지
#
# 실행:
#   python HUBROOM_V9_평가.PY --model xgboost_수치형_V9.pkl --data .\OUT_6
#   python HUBROOM_V9_평가.PY --model xgboost_수치형_V9.pkl --data .\OUT_6 ^
#       --alerts .\out_ml\밀림_경보요약.csv     ← 판정 컬럼 있는 것 (밀림요약 --episodes 산출)
# ============================================================


def load_v9_module():
    """같은 폴더의 HUBROOM_V9(.PY/.py) 를 불러와 피처 함수 재사용 (학습과 100% 동일 보장)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ('HUBROOM_V9.PY', 'HUBROOM_V9.py'):
        fp = os.path.join(here, name)
        if os.path.exists(fp):
            # ★ 대문자 .PY 도 로드되게 로더 명시 (importlib 은 소문자 .py 만 자동 인식)
            loader = importlib.machinery.SourceFileLoader('HUBROOM_V9', fp)
            spec = importlib.util.spec_from_loader('HUBROOM_V9', loader)
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)
            return mod
    raise SystemExit('❌ HUBROOM_V9.PY 가 같은 폴더에 있어야 합니다 (피처 생성 공유)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='xgboost_수치형_V9.pkl')
    ap.add_argument('--data', required=True, help='평가 원본 폴더 (예: .\\OUT_6 = 6월 일별)')
    ap.add_argument('--alerts', default=None,
                    help='(옵션) 판정 채워진 밀림_경보요약.csv → 오탐필터 성능 산출')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--out', default='V9_평가결과.csv')
    a = ap.parse_args()

    v9 = load_v9_module()

    print('=' * 80)
    print('🎯 V9 평가 — 홀드아웃(학습 안 본 기간)')
    print('=' * 80)

    with open(a.model, 'rb') as f:
        md = pickle.load(f)
    models, feat_names = md['models'], md['feature_names']
    print(f"모델: {a.model} (V{md.get('version','9')}, 피처 {len(feat_names)}개)")

    print('\n[STEP 1] 평가 원본 로드')
    df = v9.load_raw(a.data)
    df[v9.TARGET_COL] = pd.to_numeric(df[v9.TARGET_COL], errors='coerce')
    df = df.dropna(subset=[v9.TARGET_COL]).reset_index(drop=True)

    print('\n[STEP 2] Feature 생성 (학습과 동일 코드)')
    X, y, idx = v9.create_features_v9(df, set(df.columns), stride=a.stride)
    # 학습 때와 피처 순서/구성 맞춤 (없으면 0)
    for c in feat_names:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feat_names]
    times = df['CRT_TM'].iloc[idx].dt.strftime('%Y-%m-%d %H:%M').values
    print(f'샘플 {len(X)}개')

    print('\n[STEP 3] 예측')
    pred = np.column_stack([m.predict(X) for m in models])

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    print('\n📊 ① 회귀 성능 (10분후 변화량):')
    for j, name in enumerate(['MIN', 'MAX', 'AVG']):
        mae = mean_absolute_error(y[:, j], pred[:, j])
        rmse = np.sqrt(mean_squared_error(y[:, j], pred[:, j]))
        r2 = r2_score(y[:, j], pred[:, j])
        print(f'  {name} - MAE {mae:.2f} · RMSE {rmse:.2f} · R² {r2:.3f}')

    print(f'\n🎯 ② 급등(+{v9.BIG_CHANGE}↑) 방향 예측 — 컷별:')
    act = y[:, 1] >= v9.BIG_CHANGE
    print(f'  실제 급등: {act.sum()}분 ({act.mean()*100:.1f}%)')
    for thr in [10, 20, 30, 40, 50]:
        p = pred[:, 1] >= thr
        tp = (p & act).sum()
        prec = tp / max(p.sum(), 1) * 100
        rec = tp / max(act.sum(), 1) * 100
        print(f'  예측MAX≥{thr:2d}: 정밀도 {prec:5.1f}% · 재현율 {rec:5.1f}%  (예측 {p.sum()}분)')

    # 매분 예측 저장 (연동/분석용)
    out_df = pd.DataFrame({'datetime': times,
                           '현재큐': X['target_last_value'].values,
                           '예측_MAX변화': pred[:, 1].round(1),
                           '예측_AVG변화': pred[:, 2].round(1),
                           '실제_MAX변화': y[:, 1].round(1)})
    out_df.to_csv(a.out, index=False, encoding='utf-8-sig')
    print(f'\n💾 매분 예측 → {a.out}')

    # ③ 오탐필터 성능 (경보요약 + 판정 대조)
    if a.alerts:
        print('\n' + '=' * 80)
        print('🎯 ③ CUSUM 경보 오탐필터 성능 (판정표 대조)')
        print('=' * 80)
        al = pd.read_csv(a.alerts, encoding='utf-8-sig')
        if '판정' not in al.columns:
            print('⚠️ 판정 컬럼 없음 — 밀림요약.py --episodes 로 만든 파일을 주세요')
            return
        pmap = dict(zip(out_df['datetime'], out_df['예측_MAX변화']))
        rows = []
        for _, r in al.iterrows():
            t = str(r['경보시작'])[:16].replace('/', '-')
            # 분 단위 표기 보정 (예: '2026-06-01 9:07' → '2026-06-01 09:07')
            d, hm = t.split(' ')
            h, m = hm.split(':')
            key = f'{d} {int(h):02d}:{int(m):02d}'
            if key in pmap:
                rows.append((r['판정'], pmap[key]))
        if not rows:
            print('⚠️ 경보시작 시각과 예측 시각 매칭 실패'); return
        jud = pd.DataFrame(rows, columns=['판정', '예측MAX'])
        n_tp = (jud['판정'] == '예측').sum()
        n_fp = (jud['판정'] == '오탐').sum()
        print(f'경보 {len(jud)}건 매칭 (정탐 {n_tp} / 오탐 {n_fp})')
        print(f"\n  {'필터 컷':^12}{'정탐 보존':^12}{'오탐 제거':^12}  ← 컷 이상만 경보 유지 시")
        for thr in [0, 5, 10, 15, 20, 30]:
            keep = jud['예측MAX'] >= thr
            tp_keep = ((jud['판정'] == '예측') & keep).sum() / max(n_tp, 1) * 100
            fp_kill = ((jud['판정'] == '오탐') & ~keep).sum() / max(n_fp, 1) * 100
            print(f'  예측MAX≥{thr:2d}   {tp_keep:6.0f}%      {fp_kill:6.0f}%')
        print('\n  → 정탐보존 90%↑ 유지하면서 오탐제거 최대인 컷을 고르면 됨')

    print('\n🎉 V9 평가 완료!')


if __name__ == '__main__':
    main()

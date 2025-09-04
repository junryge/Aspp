# -*- coding: utf-8 -*-
"""
PatchTST & PatchTST+PINN 모델 구축을 위한 실제 데이터 검증 (개선된 버전)
- 컬럼별 상세 평가 및 모델별 등급/점수 시스템 추가
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 컬럼 평가 기준 정의
COLUMN_DEFINITIONS = [
    # 타겟
    {"컬럼명": "CURRENT_M16A_3F_JOB_2", "카테고리": "타겟", "V4필수도": "필수", "점수": 100, "물리법칙": "핵심"},
    # 유입
    {"컬럼명": "M16A_6F_TO_HUB_JOB", "카테고리": "유입", "V4필수도": "필수", "점수": 70, "물리법칙": "핵심"},
    {"컬럼명": "M16A_2F_TO_HUB_JOB2", "카테고리": "유입", "V4필수도": "필수", "점수": 65, "물리법칙": "핵심"},
    {"컬럼명": "M14A_3F_TO_HUB_JOB2", "카테고리": "유입", "V4필수도": "필수", "점수": 65, "물리법칙": "핵심"},
    {"컬럼명": "M14B_7F_TO_HUB_JOB2", "카테고리": "유입", "V4필수도": "필수", "점수": 65, "물리법칙": "핵심"},
    {"컬럼명": "M16B_10F_TO_HUB_JOB", "카테고리": "유입", "V4필수도": "선택", "점수": 25, "물리법칙": "보조"},
    # 유출
    {"컬럼명": "M16A_3F_TO_M16A_6F_JOB", "카테고리": "유출", "V4필수도": "필수", "점수": 60, "물리법칙": "핵심"},
    {"컬럼명": "M16A_3F_TO_M16A_2F_JOB", "카테고리": "유출", "V4필수도": "필수", "점수": 55, "물리법칙": "핵심"},
    {"컬럼명": "M16A_3F_TO_M14A_3F_JOB", "카테고리": "유출", "V4필수도": "필수", "점수": 55, "물리법칙": "핵심"},
    {"컬럼명": "M16A_3F_TO_M14B_7F_JOB", "카테고리": "유출", "V4필수도": "필수", "점수": 55, "물리법칙": "핵심"},
    {"컬럼명": "M16A_3F_TO_3F_MLUD_JOB", "카테고리": "유출", "V4필수도": "선택", "점수": 35, "물리법칙": "보조"},
    # CMD
    {"컬럼명": "M16A_3F_CMD", "카테고리": "내부CMD", "V4필수도": "필수", "점수": 85, "물리법칙": "핵심"},
    {"컬럼명": "M16A_6F_TO_HUB_CMD", "카테고리": "외부CMD", "V4필수도": "권장", "점수": 55, "물리법칙": "보조"},
    {"컬럼명": "M16A_2F_TO_HUB_CMD", "카테고리": "외부CMD", "V4필수도": "선택", "점수": 50, "물리법칙": "보조"},
    {"컬럼명": "M14A_3F_TO_HUB_CMD", "카테고리": "외부CMD", "V4필수도": "선택", "점수": 50, "물리법칙": "보조"},
    {"컬럼명": "M14B_7F_TO_HUB_CMD", "카테고리": "외부CMD", "V4필수도": "선택", "점수": 50, "물리법칙": "보조"},
    # MaxCapa
    {"컬럼명": "M16A_6F_LFT_MAXCAPA", "카테고리": "MaxCapa", "V4필수도": "선택", "점수": 50, "물리법칙": "보조"},
    {"컬럼명": "M16A_2F_LFT_MAXCAPA", "카테고리": "MaxCapa", "V4필수도": "선택", "점수": 45, "물리법칙": "보조"},
    # 극단값지표
    {"컬럼명": "M16A_3F_STORAGE_UTIL", "카테고리": "극단값지표", "V4필수도": "필수", "점수": 95, "물리법칙": "핵심"},
    # OFS
    {"컬럼명": "M14_TO_M16_OFS_CUR", "카테고리": "OFS", "V4필수도": "선택", "점수": 40, "물리법칙": "보조"},
    {"컬럼명": "M16_TO_M14_OFS_CUR", "카테고리": "OFS", "V4필수도": "선택", "점수": 40, "물리법칙": "보조"},
    # 물리 지표
    {"컬럼명": "BRIDGE_TIME", "카테고리": "물리지표", "V4필수도": "필수", "점수": 85, "물리법칙": "핵심"},
]

def get_grade(score, max_score):
    """점수를 기준으로 등급을 반환하는 함수"""
    if max_score == 0:
        return 'N/A', '⚫'
    ratio = score / max_score
    if ratio >= 0.9:
        return 'A+', '🟢'
    elif ratio >= 0.8:
        return 'A', '🟢'
    elif ratio >= 0.7:
        return 'B+', '🟡'
    elif ratio >= 0.6:
        return 'B', '🟡'
    elif ratio >= 0.4:
        return 'C', '🟠'
    else:
        return 'F', '🔴'

def validate_real_data_upgraded():
    """
    컬럼별 상세 평가 및 모델별 등급 시스템이 추가된 데이터 검증 함수
    """
    print("=" * 80)
    print("🔍 PatchTST & PatchTST+PINN 모델 데이터 종합 검증")
    print(f"📅 검증 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. 데이터 로드 (기존 코드와 동일)
    data_main = 'data/HUB_0509_TO_0807_DATA.CSV'
    data_bridge = 'data/BRTIME_0509_TO_0831.CSV'

    if not (os.path.exists(data_main) and os.path.exists(data_bridge)):
        print("\n❌ 데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return
        
    df_main = pd.read_csv(data_main, encoding='utf-8-sig')
    df_bridge = pd.read_csv(data_bridge, encoding='utf-8-sig')
    
    # 2. 시간 컬럼 처리 및 데이터 병합 (기존 코드와 동일)
    df_main['time'] = pd.to_datetime(df_main['STAT_DT'].astype(str), format='%Y%m%d%H%M')
    df_bridge['time'] = pd.to_datetime(df_bridge['CRT_TM'].str[:19])
    df = pd.merge(df_main, df_bridge[['time', 'IDC_VAL']], on='time', how='left')
    df.rename(columns={'IDC_VAL': 'BRIDGE_TIME'}, inplace=True)
    df['BRIDGE_TIME'] = df['BRIDGE_TIME'].interpolate(method='linear').fillna(df['BRIDGE_TIME'].mean())
    
    print(f"\n✅ 데이터 로드 및 병합 완료: {df.shape[0]:,}행 × {df.shape[1]}열")
    
    # 3. 컬럼별 상세 평가 (신규)
    print("\n" + "=" * 80)
    print("📋 1. 컬럼별 상세 평가 리포트")
    print("-" * 80)
    
    eval_df = pd.DataFrame(COLUMN_DEFINITIONS)
    eval_df['존재여부'] = eval_df['컬럼명'].apply(lambda x: '✅' if x in df.columns else '❌')
    
    # 카테고리별로 그룹화하여 출력
    for category, group in eval_df.groupby('카테고리'):
        print(f"\n--- {category} ({len(group)}개) ---")
        for _, row in group.iterrows():
            print(f"{row['존재여부']} {row['컬럼명'].ljust(30)} | 필수도: {row['V4필수도'].ljust(4)} | 점수: {row['점수']}")

    # 4. 모델별 구축 가능성 평가 (개선)
    print("\n" + "=" * 80)
    print("🤖 2. 모델별 준비도 및 등급 평가")
    print("-" * 80)
    
    # 모델별 평가에 사용될 카테고리 정의
    pinn_categories = ['타겟', '유입', '유출', '물리지표']
    patchtst_categories = ['타겟', '유입', '유출', '내부CMD', '외부CMD', 'MaxCapa', '극단값지표', '물리지표', 'OFS']

    # --- PatchTST+PINN 평가 ---
    pinn_eval = eval_df[eval_df['카테고리'].isin(pinn_categories)]
    pinn_max_score = pinn_eval['점수'].sum()
    pinn_current_score = pinn_eval[pinn_eval['존재여부'] == '✅']['점수'].sum()
    pinn_grade, pinn_emoji = get_grade(pinn_current_score, pinn_max_score)
    
    print(f"\n{pinn_emoji} Model 1: PatchTST + PINN (물리법칙 기반 모델)")
    print(f"   - 평가 점수: {pinn_current_score} / {pinn_max_score}")
    print(f"   - 준비도 등급: {pinn_grade}")
    
    # PINN 구축을 위한 핵심 조건 확인
    has_target = '✅' in eval_df[eval_df['카테고리'] == '타겟']['존재여부'].values
    has_inflow = '✅' in eval_df[eval_df['카테고리'] == '유입']['존재여부'].values
    has_outflow = '✅' in eval_df[eval_df['카테고리'] == '유출']['존재여부'].values
    
    if has_target and has_inflow and has_outflow:
        print("   - 구축 여부: ✅ 가능 (물리법칙 적용을 위한 핵심 컬럼 모두 존재)")
    else:
        print("   - 구축 여부: ❌ 불가 (타겟, 유입, 유출 중 하나 이상 누락)")

    # --- PatchTST 평가 ---
    patchtst_eval = eval_df[eval_df['카테고리'].isin(patchtst_categories)]
    patchtst_max_score = patchtst_eval['점수'].sum()
    patchtst_current_score = patchtst_eval[patchtst_eval['존재여부'] == '✅']['점수'].sum()
    patchtst_grade, patchtst_emoji = get_grade(patchtst_current_score, patchtst_max_score)

    print(f"\n{patchtst_emoji} Model 2: PatchTST (종합 예측 모델)")
    print(f"   - 평가 점수: {patchtst_current_score} / {patchtst_max_score}")
    print(f"   - 준비도 등급: {patchtst_grade}")
    if has_target:
        print("   - 구축 여부: ✅ 가능 (타겟 컬럼 존재)")
    else:
        print("   - 구축 여부: ❌ 불가 (타겟 컬럼 누락)")
        
    # 5. 물리법칙 및 데이터 품질 검증 (기존 코드 유지)
    print("\n" + "=" * 80)
    print("⚖️ 3. 물리법칙 및 데이터 품질 검증")
    print("-" * 80)

    # 물리법칙 균형
    inflow_cols = eval_df[(eval_df['카테고리'] == '유입') & (eval_df['존재여부'] == '✅')]['컬럼명'].tolist()
    outflow_cols = eval_df[(eval_df['카테고리'] == '유출') & (eval_df['존재여부'] == '✅')]['컬럼명'].tolist()
    
    if has_target and inflow_cols and outflow_cols:
        stock_change = df['CURRENT_M16A_3F_JOB_2'].diff()
        balance_error = (df[inflow_cols].sum(axis=1) - df[outflow_cols].sum(axis=1) - stock_change).abs().mean()
        print(f"📊 물리법칙 균형 오차: {balance_error:.2f} (낮을수록 좋음)")
        if balance_error < 10: print("   - 평가: ✅ 우수 (물리적 관계가 데이터에 잘 반영됨)")
        else: print("   - 평가: ⚠️ 양호 (약간의 불균형 존재, 모델이 보정 가능)")
    else:
        print("📊 물리법칙 균형: ❌ 검증 불가 (핵심 컬럼 누락)")

    # 브리지타임 상관관계
    if 'BRIDGE_TIME' in df.columns and has_target:
        corr = df[['CURRENT_M16A_3F_JOB_2', 'BRIDGE_TIME']].corr().iloc[0, 1]
        print(f"🔗 브리지타임-타겟 상관관계: {corr:.3f}")

    # 시계열 품질
    time_diff = df['time'].diff().dt.total_seconds().mode()[0] / 60
    print(f"📈 데이터 간격: {time_diff:.0f}분")
    print(f"   - 총 기간: {(df['time'].max() - df['time'].min()).days}일")
    print(f"   - 학습 샘플 수: {len(df) - 30:,}개 (20분 입력, 10분 예측 기준)")

    # 6. 최종 권고사항 (개선)
    print("\n" + "=" * 80)
    print("💡 4. 최종 요약 및 권고사항")
    print("-" * 80)
    
    print(f"현재 데이터셋은 **PatchTST+PINN 모델 기준 {pinn_grade} 등급**, **PatchTST 모델 기준 {patchtst_grade} 등급**의 준비도를 보입니다.\n")

    if pinn_grade in ['A+', 'A'] and (has_target and has_inflow and has_outflow):
        print("🥇 **최우선 권장: PatchTST+PINN 모델 구축**")
        print("   - 물리법칙을 적용하기 위한 핵심 데이터가 모두 확보되어 모델의 안정성과 정확도를 크게 높일 수 있습니다.")
        print("   - 재고량 예측의 신뢰도를 극대화할 수 있는 최적의 조건입니다.")
    elif patchtst_grade in ['A+', 'A', 'B+']:
        print("🥈 **차선 권장: PatchTST 모델 우선 구축**")
        print("   - 종합적인 예측 성능을 내기에 충분한 데이터가 확보되었습니다.")
        print(f"   - {'물리법칙 관련 데이터(유입/유출) 보강 시 PINN 모델로 확장 가능합니다.' if not (has_inflow and has_outflow) else 'PINN 모델도 병행하여 성능을 비교하는 것을 추천합니다.'}")
    else:
        print("🔴 **주의: 데이터 보강 필요**")
        print("   - 현재 데이터로는 모델의 성능을 보장하기 어렵습니다.")
        print("   - 아래 '❌'로 표시된, 점수가 높은 필수 컬럼들을 우선적으로 확보해야 합니다.")
        missing_critical = eval_df[(eval_df['존재여부'] == '❌') & (eval_df['V4필수도'].isin(['필수', '권장']))]
        if not missing_critical.empty:
            print("   - **[시급한 데이터]** " + ", ".join(missing_critical['컬럼명'].tolist()))

    print("\n" + "="*80)
    print("🏁 검증 완료!")
    print("="*80)

# 실행
if __name__ == "__main__":
    validate_real_data_upgraded()

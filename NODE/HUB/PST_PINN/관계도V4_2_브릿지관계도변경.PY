# -*- coding: utf-8 -*-
"""
PatchTST & PatchTST+PINN 모델 구축을 위한 실제 데이터 검증
실제 파일: HUB_0509_TO_0807_DATA.CSV, BRTIME_0509_TO_0831.CSV
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 그래프 한글 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def validate_real_data():
    """
    실제 데이터 검증
    Model 1: PatchTST (전체 균형)
    Model 2: PatchTST + PINN (물리법칙 적용)
    """
    
    print("=" * 80)
    print("🔍 PatchTST & PatchTST+PINN 모델 데이터 검증")
    print(f"📅 검증 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 실제 데이터 파일 경로
    data_main = 'data/HUB_0509_TO_0807_DATA.CSV'
    data_bridge = 'data/BRTIME_0509_TO_0831.CSV'
    
    # 1. 파일 존재 확인
    print("\n📁 데이터 파일 확인:")
    print("-" * 60)
    
    if not os.path.exists(data_main):
        print(f"❌ {data_main} 파일을 찾을 수 없습니다.")
        return None
        
    if not os.path.exists(data_bridge):
        print(f"❌ {data_bridge} 파일을 찾을 수 없습니다.")
        return None
    
    # 2. 데이터 로드
    print("\n📊 데이터 로드:")
    print("-" * 60)
    
    df_main = pd.read_csv(data_main, encoding='utf-8-sig')
    df_bridge = pd.read_csv(data_bridge, encoding='utf-8-sig')
    
    print(f"✅ HUB 데이터: {df_main.shape[0]:,}행 × {df_main.shape[1]}열")
    print(f"✅ 브리지타임 데이터: {df_bridge.shape[0]:,}행 × {df_bridge.shape[1]}열")
    
    # 3. 타겟 컬럼 확인
    print("\n🎯 타겟 컬럼 확인:")
    print("-" * 60)
    
    target_col = 'CURRENT_M16A_3F_JOB_2'
    if target_col in df_main.columns:
        print(f"✅ 타겟 컬럼 '{target_col}' 존재")
        print(f"   - 평균: {df_main[target_col].mean():.2f}")
        print(f"   - 표준편차: {df_main[target_col].std():.2f}")
        print(f"   - 최소값: {df_main[target_col].min():.2f}")
        print(f"   - 최대값: {df_main[target_col].max():.2f}")
        print(f"   - 결측치: {df_main[target_col].isna().sum()}개")
    else:
        print(f"❌ 타겟 컬럼 '{target_col}' 없음")
        return None
    
    # 4. 시간 컬럼 확인 및 병합
    print("\n⏰ 시간 정보 및 병합:")
    print("-" * 60)
    
    # 시간 컬럼 확인
    time_col_main = None
    time_col_bridge = None
    
    # HUB 데이터 시간 컬럼 찾기
    for col in ['STAT_DT', 'DATETIME', 'TIME', 'DATE']:
        if col in df_main.columns:
            time_col_main = col
            break
    
    # 브리지타임 시간 컬럼 찾기
    for col in ['CRT_TM', 'DATETIME', 'TIME', 'DATE']:
        if col in df_bridge.columns:
            time_col_bridge = col
            break
    
    if time_col_main:
        print(f"✅ HUB 시간 컬럼: {time_col_main}")
    else:
        print("❌ HUB 데이터에서 시간 컬럼을 찾을 수 없음")
    
    if time_col_bridge:
        print(f"✅ 브리지타임 시간 컬럼: {time_col_bridge}")
    else:
        print("❌ 브리지타임 데이터에서 시간 컬럼을 찾을 수 없음")
    
    # 시간 변환 및 병합
    if time_col_main and time_col_bridge:
        try:
            # HUB 데이터 시간 변환
            if time_col_main == 'STAT_DT':
                df_main['time'] = pd.to_datetime(df_main[time_col_main].astype(str), format='%Y%m%d%H%M')
            else:
                df_main['time'] = pd.to_datetime(df_main[time_col_main])
            
            # 브리지타임 시간 변환
            if time_col_bridge == 'CRT_TM':
                df_bridge['time'] = pd.to_datetime(df_bridge[time_col_bridge].str[:19])
            else:
                df_bridge['time'] = pd.to_datetime(df_bridge[time_col_bridge])
            
            # 병합
            bridge_value_col = 'IDC_VAL' if 'IDC_VAL' in df_bridge.columns else df_bridge.columns[1]
            df = pd.merge(df_main, df_bridge[['time', bridge_value_col]], 
                         on='time', how='left')
            df.rename(columns={bridge_value_col: 'BRIDGE_TIME'}, inplace=True)
            
            # 브리지타임 결측치 처리
            bridge_missing_before = df['BRIDGE_TIME'].isna().sum()
            df['BRIDGE_TIME'] = df['BRIDGE_TIME'].interpolate(method='linear')
            df['BRIDGE_TIME'] = df['BRIDGE_TIME'].fillna(df['BRIDGE_TIME'].mean())
            
            print(f"✅ 데이터 병합 완료: {df.shape[0]:,}행 × {df.shape[1]}열")
            print(f"   브리지타임 결측치: {bridge_missing_before}개 → 0개 (보간 완료)")
        except Exception as e:
            print(f"❌ 시간 변환/병합 중 오류: {e}")
            df = df_main.copy()
    else:
        df = df_main.copy()
    
    # 5. 필수 컬럼 분석
    print("\n📋 필수 컬럼 분석:")
    print("-" * 60)
    
    # 컬럼 카테고리 정의
    column_categories = {
        '타겟': ['CURRENT_M16A_3F_JOB_2'],
        '유입_필수': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 
                   'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2'],
        '유출_필수': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
                   'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB'],
        'CMD_핵심': ['M16A_3F_CMD'],
        'Storage': ['M16A_3F_STORAGE_UTIL'],
        '브리지타임': ['BRIDGE_TIME'],
        '활용률': ['M16A_6F_LFT_MAXCAPA_UTIL', 'M16A_2F_LFT_MAXCAPA_UTIL',
                 'M14A_3F_CNV_MAXCAPA_UTIL', 'M14B_7F_LFT_MAXCAPA_UTIL'],
        'MaxCapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA',
                   'M14A_3F_CNV_MAXCAPA', 'M14B_7F_LFT_MAXCAPA'],
        'OFS': ['M14_TO_M16_OFS_CUR', 'M16_TO_M14_OFS_CUR'],
        '집계': ['M16_ALL_TO_HUB_JOB', 'M14_ALL_TO_HUB_JOB', 'ALL_TO_HUB_JOB']
    }
    
    # 각 카테고리별 존재 여부 확인
    validation_results = {}
    total_score = 0
    available_score = 0
    
    for category, cols in column_categories.items():
        validation_results[category] = {
            'total': len(cols),
            'found': 0,
            'missing': [],
            'existing': []
        }
        
        for col in cols:
            if col in df.columns:
                validation_results[category]['found'] += 1
                validation_results[category]['existing'].append(col)
                # 점수 계산 (카테고리별 가중치)
                if category == '타겟':
                    available_score += 100
                elif category in ['유입_필수', '유출_필수']:
                    available_score += 60
                elif category in ['CMD_핵심', 'Storage', '브리지타임']:
                    available_score += 80
                else:
                    available_score += 40
            else:
                validation_results[category]['missing'].append(col)
            
            # 총점 계산
            if category == '타겟':
                total_score += 100
            elif category in ['유입_필수', '유출_필수']:
                total_score += 60
            elif category in ['CMD_핵심', 'Storage', '브리지타임']:
                total_score += 80
            else:
                total_score += 40
        
        # 카테고리별 출력
        found_ratio = validation_results[category]['found'] / validation_results[category]['total'] * 100
        status = "✅" if found_ratio == 100 else "⚠️" if found_ratio >= 50 else "❌"
        
        print(f"\n{status} {category}: {validation_results[category]['found']}/{validation_results[category]['total']} ({found_ratio:.0f}%)")
        
        if validation_results[category]['existing']:
            for col in validation_results[category]['existing'][:3]:  # 최대 3개만 표시
                print(f"   ✓ {col}")
            if len(validation_results[category]['existing']) > 3:
                print(f"   ... 외 {len(validation_results[category]['existing'])-3}개")
        
        if validation_results[category]['missing']:
            for col in validation_results[category]['missing'][:3]:  # 최대 3개만 표시
                print(f"   ✗ {col}")
            if len(validation_results[category]['missing']) > 3:
                print(f"   ... 외 {len(validation_results[category]['missing'])-3}개")
    
    # 6. 물리법칙 검증 (유입-유출 균형)
    print("\n⚖️ 물리법칙 검증:")
    print("-" * 60)
    
    # 유입 컬럼 합계
    inflow_cols = [col for col in validation_results['유입_필수']['existing']]
    if inflow_cols:
        total_inflow = df[inflow_cols].sum(axis=1)
        print(f"✅ 총 유입 계산 가능 ({len(inflow_cols)}개 컬럼)")
    else:
        total_inflow = 0
        print("❌ 유입 컬럼 부족으로 계산 불가")
    
    # 유출 컬럼 합계
    outflow_cols = [col for col in validation_results['유출_필수']['existing']]
    if outflow_cols:
        total_outflow = df[outflow_cols].sum(axis=1)
        print(f"✅ 총 유출 계산 가능 ({len(outflow_cols)}개 컬럼)")
    else:
        total_outflow = 0
        print("❌ 유출 컬럼 부족으로 계산 불가")
    
    # 재고 변화량
    if target_col in df.columns:
        stock_change = df[target_col].diff()
        
        # 물리법칙: 재고 변화 = 유입 - 유출
        if isinstance(total_inflow, pd.Series) and isinstance(total_outflow, pd.Series):
            balance = total_inflow - total_outflow - stock_change
            balance_error = balance.abs().mean()
            print(f"\n📊 물리법칙 균형 오차: {balance_error:.2f}")
            
            if balance_error < 10:
                print("   ✅ 우수: 물리법칙 잘 유지됨")
            elif balance_error < 50:
                print("   ⚠️ 양호: 약간의 불균형 존재")
            else:
                print("   ❌ 주의: 물리법칙 불균형 심각")
    
    # 7. 브리지타임 상관관계 분석
    print("\n🔗 브리지타임 상관관계:")
    print("-" * 60)
    
    if 'BRIDGE_TIME' in df.columns and target_col in df.columns:
        # 동시점 상관관계
        corr_instant = df[[target_col, 'BRIDGE_TIME']].corr().iloc[0, 1]
        print(f"동시점 상관계수: {corr_instant:.3f}")
        
        # 10분 후 예측을 위한 lag 상관관계
        df['TARGET_LAG10'] = df[target_col].shift(-10)
        corr_lag10 = df[['BRIDGE_TIME', 'TARGET_LAG10']].corr().iloc[0, 1]
        print(f"10분 예측 상관계수: {corr_lag10:.3f}")
        
        # 다양한 lag 분석
        lag_corrs = {}
        for lag in [5, 10, 15, 20]:
            df[f'BRIDGE_LAG{lag}'] = df['BRIDGE_TIME'].shift(lag)
            lag_corr = df[[f'BRIDGE_LAG{lag}', 'TARGET_LAG10']].corr().iloc[0, 1]
            lag_corrs[lag] = lag_corr
            print(f"Bridge(t-{lag}) → Target(t+10) 상관계수: {lag_corr:.3f}")
        
        best_lag = max(lag_corrs, key=lambda x: abs(lag_corrs[x]))
        print(f"\n✨ 최적 Lag: {best_lag}분 (상관계수: {lag_corrs[best_lag]:.3f})")
    else:
        print("❌ 브리지타임 데이터 없음")
    
    # 8. 시계열 데이터 품질 검증
    print("\n📈 시계열 데이터 품질:")
    print("-" * 60)
    
    if 'time' in df.columns:
        # 시간 간격 분석
        time_diff = df['time'].diff().dt.total_seconds() / 60  # 분 단위
        time_interval = time_diff.mode().values[0] if len(time_diff.mode()) > 0 else time_diff.median()
        
        print(f"데이터 간격: {time_interval:.0f}분")
        print(f"총 기간: {df['time'].min().date()} ~ {df['time'].max().date()}")
        print(f"총 일수: {(df['time'].max() - df['time'].min()).days}일")
        
        # 결측 구간 확인
        missing_intervals = time_diff[time_diff > time_interval * 2].count()
        print(f"결측 구간: {missing_intervals}개")
        
        # 20분 윈도우로 10분 예측 가능 샘플 수
        valid_samples = len(df) - 30  # 20분 입력 + 10분 후 타겟
        print(f"학습 가능 샘플: {valid_samples:,}개")
        
        if valid_samples > 10000:
            print("   ✅ 충분한 학습 데이터")
        elif valid_samples > 5000:
            print("   ⚠️ 적절한 학습 데이터")
        else:
            print("   ❌ 학습 데이터 부족")
    
    # 9. 모델별 구축 가능성 평가
    print("\n🤖 모델 구축 가능성 평가:")
    print("-" * 60)
    
    score_ratio = (available_score / total_score * 100) if total_score > 0 else 0
    
    # 필수 조건 확인
    has_target = validation_results['타겟']['found'] > 0
    has_inflow = validation_results['유입_필수']['found'] >= 2
    has_outflow = validation_results['유출_필수']['found'] >= 2
    has_bridge = validation_results['브리지타임']['found'] > 0
    
    print(f"\n📊 데이터 준비도: {score_ratio:.1f}%")
    print(f"   • 타겟: {'✅' if has_target else '❌'}")
    print(f"   • 유입: {'✅' if has_inflow else '❌'} ({validation_results['유입_필수']['found']}/4)")
    print(f"   • 유출: {'✅' if has_outflow else '❌'} ({validation_results['유출_필수']['found']}/4)")
    print(f"   • 브리지타임: {'✅' if has_bridge else '❌'}")
    
    # Model 1: PatchTST 평가
    print("\n📍 Model 1: PatchTST (전체 균형)")
    if has_target and (has_inflow or has_outflow):
        print("   ✅ 구축 가능")
        print("   - 기본 시계열 패턴 학습 가능")
        print("   - 브리지타임으로 성능 향상 가능" if has_bridge else "   - 브리지타임 없이도 구축 가능")
    else:
        print("   ❌ 구축 불가")
        print("   - 최소한 타겟과 유입/유출 데이터 필요")
    
    # Model 2: PatchTST + PINN 평가
    print("\n📍 Model 2: PatchTST + PINN (물리법칙)")
    if has_target and has_inflow and has_outflow:
        print("   ✅ 구축 가능")
        print("   - 물리법칙 (유입-유출 균형) 적용 가능")
        print("   - 브리지타임으로 추가 제약조건 가능" if has_bridge else "   - 기본 물리법칙만 적용")
    else:
        print("   ⚠️ 제한적 구축")
        print("   - 유입/유출 데이터 부족으로 물리법칙 제약")
    
    # 10. 최종 권고사항
    print("\n💡 권고사항:")
    print("-" * 60)
    
    if score_ratio >= 70:
        print("✅ 즉시 모델 구축 가능")
        print("   1. 두 모델 모두 구축하여 성능 비교")
        print("   2. 브리지타임 lag 최적화 수행")
        print("   3. 물리법칙 가중치 튜닝")
    elif score_ratio >= 50:
        print("⚠️ 기본 모델 구축 가능")
        print("   1. PatchTST 모델 우선 구축")
        print("   2. 누락 데이터 수집 후 PINN 추가")
        print("   3. 브리지타임 활용 극대화")
    else:
        print("❌ 추가 데이터 수집 필요")
        print("   1. 유입/유출 컬럼 확보 우선")
        print("   2. 브리지타임 데이터 품질 개선")
        print("   3. 최소 1개월 이상 데이터 축적")
    
    # 결과 저장
    validation_summary = {
        '데이터준비도': f"{score_ratio:.1f}%",
        '타겟존재': has_target,
        '유입컬럼': validation_results['유입_필수']['found'],
        '유출컬럼': validation_results['유출_필수']['found'],
        '브리지타임': has_bridge,
        'PatchTST구축가능': has_target and (has_inflow or has_outflow),
        'PINN구축가능': has_target and has_inflow and has_outflow,
        '총샘플수': len(df),
        '검증시간': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # CSV로 저장
    pd.DataFrame([validation_summary]).to_csv('model_validation_result.csv', index=False, encoding='utf-8-sig')
    print("\n💾 검증 결과 저장: model_validation_result.csv")
    
    return validation_results, df

# 실행
if __name__ == "__main__":
    results, data = validate_real_data()
    print("\n" + "=" * 80)
    print("🏁 검증 완료!")
    print("=" * 80)
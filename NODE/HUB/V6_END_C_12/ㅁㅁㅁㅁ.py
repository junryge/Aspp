# -*- coding: utf-8 -*-
"""
사전감지 케이스 분석기
- 과거 30개 데이터(시퀀스)가 모두 300 미만
- 다음 시점(실제값)이 300 이상
- 이런 급격한 점프를 사전에 감지할 수 있는지 분석
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def analyze_early_detection():
    """
    평가 결과 CSV에서 사전감지 케이스 추출 및 분석
    """
    print("="*80)
    print("🔍 사전감지 케이스 분석 시작")
    print("="*80)
    
    # 1. 평가 결과 CSV 로드
    try:
        df_eval = pd.read_csv('prediction_evaluation_컬럼12_10_1013.csv', encoding='utf-8-sig')
        print(f"✅ 평가 결과 로드 완료: {len(df_eval)}개 행")
    except FileNotFoundError:
        print("❌ prediction_evaluation_컬럼12_10_1013.csv 파일을 찾을 수 없습니다.")
        print("   다른 평가 결과 파일명을 입력하세요.")
        return None
    
    # 2. 원본 데이터 로드 (시퀀스 데이터 확인용)
    try:
        df_raw = pd.read_csv('HUB0905101512.CSV', on_bad_lines='skip')
        print(f"✅ 원본 데이터 로드 완료: {len(df_raw)}개 행")
    except FileNotFoundError:
        print("❌ HUB0905101512.CSV 파일을 찾을 수 없습니다.")
        return None
    
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    # 3. 사전감지 케이스 찾기
    early_detection_cases = []
    
    print("\n🔍 사전감지 케이스 탐색 중...")
    
    for i in range(30, len(df_raw)):
        # 과거 30개 시퀀스
        seq_data = df_raw[TARGET_COL].iloc[i-30:i].values
        
        # 현재 시점 실제값
        actual_value = df_raw[TARGET_COL].iloc[i]
        
        # 사전감지 조건:
        # 1. 시퀀스 30개 모두 300 미만
        # 2. 실제값은 300 이상
        if np.all(seq_data < 300) and actual_value >= 300:
            # 해당 인덱스의 평가 결과 찾기
            idx_in_eval = i - 30  # 평가 결과는 30부터 시작했으므로
            
            if idx_in_eval < len(df_eval):
                eval_row = df_eval.iloc[idx_in_eval]
                
                early_detection_cases.append({
                    '원본인덱스': i,
                    '현재시간': eval_row['현재시간'],
                    '예측시점': eval_row['예측시점'],
                    '실제시점': eval_row['실제시점'],
                    '시퀀스MAX': round(np.max(seq_data), 2),
                    '시퀀스MIN': round(np.min(seq_data), 2),
                    '시퀀스평균': round(np.mean(seq_data), 2),
                    '시퀀스STD': round(np.std(seq_data), 2),
                    '실제값': round(actual_value, 2),
                    '예측값': round(eval_row['예측값'], 2),
                    '오차': round(actual_value - eval_row['예측값'], 2),
                    '오차율(%)': round(abs(actual_value - eval_row['예측값']) / actual_value * 100, 2),
                    '사전감지': '✅ 성공' if eval_row['예측값'] >= 290 else '❌ 실패',
                    '사전감지_점수': round(eval_row['예측값'] / actual_value * 100, 2)
                })
                
        # 진행상황 출력
        if i % 500 == 0:
            print(f"  진행중... {i}/{len(df_raw)} ({i/len(df_raw)*100:.1f}%)")
    
    # 4. DataFrame 변환
    df_early = pd.DataFrame(early_detection_cases)
    
    if len(df_early) == 0:
        print("\n⚠️ 사전감지 케이스를 찾을 수 없습니다.")
        return None
    
    print(f"\n✅ 사전감지 케이스 발견: {len(df_early)}개")
    
    # 5. 통계 분석
    print("\n" + "="*80)
    print("📊 사전감지 케이스 통계")
    print("="*80)
    
    success_count = (df_early['예측값'] >= 290).sum()
    success_rate = success_count / len(df_early) * 100
    
    print(f"총 사전감지 케이스: {len(df_early)}개")
    print(f"사전감지 성공 (예측값≥290): {success_count}개 ({success_rate:.1f}%)")
    print(f"사전감지 실패 (예측값<290): {len(df_early) - success_count}개 ({100-success_rate:.1f}%)")
    print(f"\n평균 오차: {df_early['오차'].abs().mean():.2f}")
    print(f"평균 오차율: {df_early['오차율(%)'].mean():.2f}%")
    print(f"최대 실제값: {df_early['실제값'].max():.2f}")
    print(f"최소 실제값: {df_early['실제값'].min():.2f}")
    print(f"\n평균 시퀀스MAX: {df_early['시퀀스MAX'].mean():.2f}")
    print(f"평균 시퀀스평균: {df_early['시퀀스평균'].mean():.2f}")
    
    # 6. 상세 출력 (상위 10개)
    print("\n" + "="*80)
    print("🔥 사전감지 케이스 상세 (상위 10개)")
    print("="*80)
    
    display_cols = ['현재시간', '시퀀스MAX', '시퀀스평균', '실제값', '예측값', '오차', '사전감지']
    print(df_early[display_cols].head(10).to_string(index=False))
    
    # 7. 사전감지 실패 케이스 분석
    failed_cases = df_early[df_early['예측값'] < 290]
    if len(failed_cases) > 0:
        print("\n" + "="*80)
        print("❌ 사전감지 실패 케이스 (예측값<290)")
        print("="*80)
        print(failed_cases[display_cols].head(10).to_string(index=False))
        
        print(f"\n실패 케이스 분석:")
        print(f"  - 평균 시퀀스MAX: {failed_cases['시퀀스MAX'].mean():.2f}")
        print(f"  - 평균 시퀀스평균: {failed_cases['시퀀스평균'].mean():.2f}")
        print(f"  - 평균 예측값: {failed_cases['예측값'].mean():.2f}")
        print(f"  - 평균 실제값: {failed_cases['실제값'].mean():.2f}")
    
    # 8. 사전감지 성공 케이스 분석
    success_cases = df_early[df_early['예측값'] >= 290]
    if len(success_cases) > 0:
        print("\n" + "="*80)
        print("✅ 사전감지 성공 케이스 (예측값≥290)")
        print("="*80)
        print(success_cases[display_cols].head(10).to_string(index=False))
        
        print(f"\n성공 케이스 분석:")
        print(f"  - 평균 시퀀스MAX: {success_cases['시퀀스MAX'].mean():.2f}")
        print(f"  - 평균 시퀀스평균: {success_cases['시퀀스평균'].mean():.2f}")
        print(f"  - 평균 예측값: {success_cases['예측값'].mean():.2f}")
        print(f"  - 평균 실제값: {success_cases['실제값'].mean():.2f}")
    
    # 9. CSV 저장
    output_file = '사전감지_분석결과.csv'
    df_early.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 사전감지 분석 결과 저장: {output_file}")
    
    # 10. 요약 레포트
    print("\n" + "="*80)
    print("📋 최종 요약 레포트")
    print("="*80)
    print(f"1. 사전감지 대상: 시퀀스30개<300 → 실제값≥300")
    print(f"2. 발견된 케이스: {len(df_early)}개")
    print(f"3. 사전감지 성공률: {success_rate:.1f}%")
    print(f"4. 평균 오차: {df_early['오차'].abs().mean():.2f}")
    print(f"5. 평균 사전감지 점수: {df_early['사전감지_점수'].mean():.1f}%")
    print(f"\n💡 인사이트:")
    
    if success_rate >= 70:
        print(f"   - 사전감지 성공률이 {success_rate:.1f}%로 우수합니다!")
        print(f"   - 모델이 급격한 점프를 잘 예측하고 있습니다.")
    elif success_rate >= 50:
        print(f"   - 사전감지 성공률이 {success_rate:.1f}%로 양호합니다.")
        print(f"   - 추가 Feature 엔지니어링으로 개선 가능합니다.")
    else:
        print(f"   - 사전감지 성공률이 {success_rate:.1f}%로 낮습니다.")
        print(f"   - 급격한 점프 예측을 위한 모델 개선이 필요합니다.")
    
    print(f"\n6. 저장 파일: {output_file}")
    
    return df_early

if __name__ == '__main__':
    print("🚀 사전감지 케이스 분석 시작...\n")
    results = analyze_early_detection()
    
    if results is not None:
        print(f"\n✅ 분석 완료! 사전감지 케이스 {len(results)}개 발견")
        print(f"📁 결과 파일: 사전감지_분석결과.csv")
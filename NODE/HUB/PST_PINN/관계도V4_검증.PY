# -*- coding: utf-8 -*-
"""
V4 버전 구축을 위한 데이터 검증 스크립트
실행하면 현재 데이터 상태와 V4 구축 가능성을 확인합니다.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def check_v4_readiness():
    """V4 구축 준비 상태 검증"""
    
    print("="*80)
    print("🔍 V4 버전 데이터 검증 시작")
    print(f"📅 검증 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 1. 데이터 파일 확인
    print("\n[Step 1] 데이터 파일 확인")
    print("-"*40)
    
    data_path = 'data/HUB_0509_TO_0730_DATA.CSV'
    
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        return False
    
    print(f"✅ 데이터 파일 존재: {data_path}")
    
    # 2. 데이터 로드 및 기본 정보
    print("\n[Step 2] 데이터 로드 및 기본 정보")
    print("-"*40)
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ 데이터 로드 성공")
        print(f"  - 전체 행 수: {len(df):,}")
        print(f"  - 전체 컬럼 수: {len(df.columns)}")
        print(f"  - 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return False
    
    # 3. V3에서 사용 중인 컬럼 확인
    print("\n[Step 3] V3 현재 사용 컬럼 확인")
    print("-"*40)
    
    v3_cols = {
        '타겟': 'CURRENT_M16A_3F_JOB_2',
        '유입1': 'M16A_6F_TO_HUB_JOB',
        '유입2': 'M16A_2F_TO_HUB_JOB2',
        '유입3': 'M14A_3F_TO_HUB_JOB2',
        '유입4': 'M14B_7F_TO_HUB_JOB2',
        '유출1': 'M16A_3F_TO_M16A_6F_JOB',
        '유출2': 'M16A_3F_TO_M16A_2F_JOB',
        '유출3': 'M16A_3F_TO_M14A_3F_JOB',
        '유출4': 'M16A_3F_TO_M14B_7F_JOB'
    }
    
    v3_missing = []
    for name, col in v3_cols.items():
        if col in df.columns:
            print(f"  ✅ {name}: {col}")
        else:
            print(f"  ❌ {name}: {col} - 없음!")
            v3_missing.append(col)
    
    if v3_missing:
        print(f"\n⚠️ 경고: V3 필수 컬럼 {len(v3_missing)}개 누락")
    
    # 4. V4 핵심 추가 컬럼 확인
    print("\n[Step 4] V4 핵심 추가 컬럼 확인")
    print("-"*40)
    
    v4_critical = {
        'M16A_3F_STORAGE_UTIL': '95%',
        'M16A_3F_CMD': '85%',
        'M16A_3F_VHL_ERR_CNT': '80%',
        'CURRENT_M16A_3F_JOB': '70%'
    }
    
    v4_status = {}
    for col, importance in v4_critical.items():
        if col in df.columns:
            print(f"  ✅ {col} (기여도 {importance}) - 있음")
            v4_status[col] = True
        else:
            print(f"  ❌ {col} (기여도 {importance}) - 없음!")
            v4_status[col] = False
    
    # 5. V4 추가 권장 컬럼 확인
    print("\n[Step 5] V4 추가 권장 컬럼 확인")
    print("-"*40)
    
    v4_recommended = [
        'M16A_6F_TO_HUB_CMD',
        'M14A_3F_TO_HUB_CMD',
        'M14B_7F_TO_HUB_CMD',
        'M16A_6F_LFT_MAXCAPA_UTIL',
        'M16A_3F_LFT_MAXCAPA_UTIL',
        'ALL_TO_HUB_JOB'
    ]
    
    recommended_count = 0
    for col in v4_recommended:
        if col in df.columns:
            print(f"  ✅ {col}")
            recommended_count += 1
        else:
            print(f"  ⭕ {col} - 선택적")
    
    print(f"\n권장 컬럼 {recommended_count}/{len(v4_recommended)}개 사용 가능")
    
    # 6. 타겟 변수 분포 분석
    print("\n[Step 6] 타겟 변수 분포 분석")
    print("-"*40)
    
    target_col = 'CURRENT_M16A_3F_JOB_2'
    
    if target_col in df.columns:
        target = df[target_col].dropna()
        
        print(f"  범위: {target.min():.0f} ~ {target.max():.0f}")
        print(f"  평균: {target.mean():.1f}")
        print(f"  표준편차: {target.std():.1f}")
        
        print("\n  극단값 분포:")
        thresholds = [300, 310, 335, 350]
        for t in thresholds:
            count = (target >= t).sum()
            pct = count / len(target) * 100
            print(f"    {t}+ : {count:,}개 ({pct:.2f}%)")
    
    # 7. 컬럼 생성 가능성 확인
    print("\n[Step 7] 대체 컬럼 생성 가능성")
    print("-"*40)
    
    # Storage 활용률 대체 가능?
    if 'M16A_3F_STORAGE_UTIL' not in df.columns:
        if target_col in df.columns:
            print("  📌 Storage 활용률 대체: CURRENT_M16A_3F_JOB_2 / 541 (최대값)")
            df['M16A_3F_STORAGE_UTIL_PROXY'] = df[target_col] / 541 * 100
            print(f"     생성된 평균값: {df['M16A_3F_STORAGE_UTIL_PROXY'].mean():.1f}%")
    
    # CMD 합계 생성 가능?
    cmd_cols = [col for col in df.columns if 'CMD' in col]
    if cmd_cols:
        print(f"  📌 CMD 관련 컬럼 {len(cmd_cols)}개 발견")
        for col in cmd_cols[:5]:  # 처음 5개만 표시
            print(f"     - {col}")
    
    # 8. 최종 V4 구축 가능성 평가
    print("\n" + "="*80)
    print("📊 V4 구축 가능성 평가")
    print("="*80)
    
    # 점수 계산
    score = 0
    max_score = 100
    
    # V3 기본 컬럼 (40점)
    v3_score = (len(v3_cols) - len(v3_missing)) / len(v3_cols) * 40
    score += v3_score
    print(f"\n1. V3 기본 컬럼: {v3_score:.1f}/40점")
    
    # V4 핵심 컬럼 (40점)
    v4_critical_score = sum(v4_status.values()) / len(v4_status) * 40
    score += v4_critical_score
    print(f"2. V4 핵심 컬럼: {v4_critical_score:.1f}/40점")
    
    # V4 권장 컬럼 (20점)
    v4_recommended_score = recommended_count / len(v4_recommended) * 20
    score += v4_recommended_score
    print(f"3. V4 권장 컬럼: {v4_recommended_score:.1f}/20점")
    
    print(f"\n총점: {score:.1f}/100점")
    
    # 등급 판정
    if score >= 80:
        grade = "A"
        message = "✅ V4 구축 준비 완료! 바로 시작 가능"
    elif score >= 60:
        grade = "B"
        message = "⚠️ 일부 컬럼 보완 후 V4 구축 가능"
    elif score >= 40:
        grade = "C"
        message = "⚠️ 핵심 컬럼 확보 필요, 대체 방안 검토"
    else:
        grade = "D"
        message = "❌ V4 구축 어려움, 데이터 보강 필수"
    
    print(f"등급: {grade}")
    print(f"판정: {message}")
    
    # 9. 권장 사항
    print("\n📝 권장 사항:")
    print("-"*40)
    
    if 'M16A_3F_STORAGE_UTIL' not in df.columns:
        print("1. M16A_3F_STORAGE_UTIL 컬럼 요청 또는 대체 생성")
    
    if 'M16A_3F_CMD' not in df.columns:
        print("2. M16A_3F_CMD 컬럼 요청 또는 CMD 관련 컬럼 합계 사용")
    
    if 'M16A_3F_VHL_ERR_CNT' not in df.columns:
        print("3. M16A_3F_VHL_ERR_CNT 컬럼 요청 (대체 어려움)")
    
    if score < 80:
        print("4. 누락된 핵심 컬럼 데이터 제공처에 요청")
        print("5. V3 모델과 앙상블 방식 고려")
    
    # 10. 실제 사용 가능한 컬럼 목록 출력
    print("\n[참고] 실제 데이터의 컬럼 목록 (처음 20개)")
    print("-"*40)
    for i, col in enumerate(df.columns[:20], 1):
        print(f"  {i:2}. {col}")
    
    if len(df.columns) > 20:
        print(f"  ... 외 {len(df.columns)-20}개")
    
    return score >= 60  # 60점 이상이면 True

if __name__ == "__main__":
    # 검증 실행
    result = check_v4_readiness()
    
    if result:
        print("\n" + "="*80)
        print("🎉 V4 구축 시작 가능!")
        print("다음 단계: V3 코드를 V4로 수정")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("⚠️ 데이터 보강 필요")
        print("위 권장사항을 참고하여 데이터 확보 후 재시도")
        print("="*80)
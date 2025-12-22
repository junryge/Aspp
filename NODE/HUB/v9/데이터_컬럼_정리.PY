#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSV 컬럼 포맷터 - 지정된 컬럼 순서로 CSV 생성
V7 모델용 데이터 추출 포맷
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ===== 타겟 컬럼 목록 (76개) =====
TARGET_COLUMNS = [
    'STAT_DT',
    'CURRENT_M16A_3F_JOB',
    'CURRENT_M16A_3F_JOB_2',
    'M14A_3F_CNV_MAXCAPA',
    'M14A_3F_TO_HUB_CMD',
    'M14A_3F_TO_HUB_JOB2',
    'M14A_3F_TO_HUB_JOB_ALT',
    'M14B_7F_LFT_MAXCAPA',
    'M14B_7F_TO_HUB_CMD',
    'M14B_7F_TO_HUB_JOB2',
    'M14B_7F_TO_HUB_JOB_ALT',
    'M14_TO_M16_OFS_CUR',
    'M16A_2F_LFT_MAXCAPA',
    'M16A_2F_TO_6F_JOB',
    'M16A_2F_TO_HUB_CMD',
    'M16A_2F_TO_HUB_JOB2',
    'M16A_2F_TO_HUB_JOB_ALT',
    'M16A_3F_CMD',
    'M16A_3F_CNV_MAXCAPA',
    'M16A_3F_LFT_MAXCAPA',
    'M16A_3F_M14BLFT_MAXCAPA',
    'M16A_3F_STORAGE_UTIL',
    'M16A_3F_TO_3F_MLUD_JOB',
    'M16A_3F_TO_M14A_3F_JOB',
    'M16A_3F_TO_M14A_CNV_AI_CMD',
    'M16A_3F_TO_M14B_7F_JOB',
    'M16A_3F_TO_M14B_LFT_AI_CMD',
    'M16A_3F_TO_M16A_2F_JOB',
    'M16A_3F_TO_M16A_3F_STB_CMD',
    'M16A_3F_TO_M16A_6F_JOB',
    'M16A_3F_TO_M16A_LFT_AI_CMD',
    'M16A_3F_TO_M16A_MLUD_AI_CMD',
    'M16A_6F_LFT_MAXCAPA',
    'M16A_6F_TO_2F_JOB',
    'M16A_6F_TO_HUB_CMD',
    'M16A_6F_TO_HUB_JOB',
    'M16A_6F_TO_HUB_JOB_ALT',
    'M16B_10F_TO_HUB_JOB',
    'M16_TO_M14_OFS_CUR',
    'HUBROOMTOTAL',
    'CD_M163FSTORAGEUSE',
    'CD_M163FSTORAGETOTAL',
    'CD_M163FSTORAGEUTIL',
    'M16HUB.QUE.ALL.CURRENTQCNT',
    'M16HUB.QUE.TIME.AVGTOTALTIME1MIN',
    'M16HUB.QUE.ALL.CURRENTQCOMPLETED',
    'M16HUB.QUE.ALL.FABTRANSJOBCNT',
    'M16HUB.QUE.TIME.AVGTOTALTIME',
    'M16HUB.QUE.OHT.CURRENTOHTQCNT',
    'M16HUB.QUE.OHT.OHTUTIL',
    'M16A.QUE.ALL.CURRENTQCOMPLETED',
    'M16A.QUE.ALL.CURRENTQCREATED',
    'M16A.QUE.OHT.CURRENTOHTQCNT',
    'M16A.QUE.OHT.OHTUTIL',
    'M16A.QUE.LOAD.AVGLOADTIME1MIN',
    'M16A.QUE.ALL.TRANSPORT4MINOVERCNT',
    'M16A.QUE.ABN.QUETIMEDELAY',
    'M16A_LFTTOALL',
    'M16HUB_M16TOM14_CREATED',
    'M16A_ALLTONORTHCNV',
    'M16A_NORTHCNVTOALL',
    'M16A_M16ATOM14A',
    'M14_TOTALCNVCURRENTQCNT',
    'M16HUB_M16TOM14B_CREATED',
    'M16HUB_M14TOM16_CREATED',
    'M16A_CURRENTQCNT',
    'M16A_ALLTOSOUTHCNV',
    'M14_ALLTOSOUTHCNV',
    'M14_ALLTONORTHCNV',
    'M14_CURRENTQCREATED',
    'M14_RETURNTOM16',
    'M14B_LFTTOALL',
    'M14B_M14BTOM16A',
    'M16B_CURRENTQCREATED',
    'M16_SENDTOM14',
    'M16HUB_M14TOM16_MESCNT',
    'M16HUB_MESCURRENTQCNT'
]

print(f"총 컬럼 수: {len(TARGET_COLUMNS)}개")


def load_csv_auto_encoding(filepath):
    """인코딩 자동 감지 CSV 로딩"""
    for enc in ['utf-8', 'cp949', 'euc-kr']:
        try:
            df = pd.read_csv(filepath, encoding=enc, on_bad_lines='skip', low_memory=False)
            print(f"[OK] {enc} 인코딩으로 로딩 성공: {len(df)}행")
            return df
        except:
            continue
    raise Exception("CSV 로딩 실패 - 인코딩 문제")


def format_csv_columns(input_file, output_file):
    """
    원본 CSV를 읽어서 지정된 컬럼 순서로 재정렬
    없는 컬럼은 NaN으로 채움
    """
    # 1. CSV 로딩
    df = load_csv_auto_encoding(input_file)
    print(f"원본 컬럼 수: {len(df.columns)}개")
    
    # 2. 컬럼 매칭 확인
    existing_cols = set(df.columns)
    target_cols = set(TARGET_COLUMNS)
    
    matched = existing_cols & target_cols
    missing = target_cols - existing_cols
    extra = existing_cols - target_cols
    
    print(f"\n[매칭 결과]")
    print(f"  - 매칭됨: {len(matched)}개")
    print(f"  - 원본에 없음: {len(missing)}개")
    print(f"  - 타겟에 없음(무시): {len(extra)}개")
    
    if missing:
        print(f"\n[경고] 원본에 없는 컬럼 (NaN으로 채움):")
        for col in sorted(missing)[:10]:
            print(f"    - {col}")
        if len(missing) > 10:
            print(f"    ... 외 {len(missing)-10}개")
    
    # 3. 타겟 컬럼 순서대로 DataFrame 구성
    result_df = pd.DataFrame()
    for col in TARGET_COLUMNS:
        if col in df.columns:
            result_df[col] = df[col]
        else:
            result_df[col] = np.nan
    
    # 4. 저장
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n[완료] 저장됨: {output_file}")
    print(f"  - 행: {len(result_df)}, 컬럼: {len(result_df.columns)}")
    
    return result_df


def create_empty_template(output_file):
    """빈 템플릿 CSV 생성 (컬럼 헤더만)"""
    df = pd.DataFrame(columns=TARGET_COLUMNS)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[완료] 빈 템플릿 생성: {output_file}")
    print(f"  - 컬럼 수: {len(TARGET_COLUMNS)}개")
    return df


def print_column_list():
    """컬럼 목록 출력 (복사용)"""
    print("\n===== 컬럼 목록 (SQL용 콤마 구분) =====")
    print(",\n".join(TARGET_COLUMNS))
    
    print("\n===== 컬럼 목록 (한 줄) =====")
    print(",".join(TARGET_COLUMNS))


# ===== 실행 예시 =====
if __name__ == "__main__":
    
    # 1. 컬럼 목록만 출력
    print_column_list()
    
    # 2. 빈 템플릿 생성
    # create_empty_template("template_76cols.csv")
    
    # 3. 기존 CSV를 컬럼 순서 맞춰서 변환
    # format_csv_columns("input.csv", "output_formatted.csv")
    
    print("\n" + "="*50)
    print("사용법:")
    print("  1. 빈 템플릿: create_empty_template('output.csv')")
    print("  2. CSV 변환: format_csv_columns('input.csv', 'output.csv')")
    print("="*50)
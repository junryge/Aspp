import pandas as pd

# 원하는 컬럼 순서
COLUMN_ORDER = [
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
    'M14.QUE.ALL.CURRENTQCOMPLETED',
    'M14.QUE.ALL.CURRENTQCREATED',
    'M14.QUE.ALL.TRANSPORT4MINOVERCNT',
    'M14.QUE.OHT.OHTUTIL',
    'M14A_A_AVG_VELOCITY',
    'M14B.QUE.SENDFAB.VERTICALQUEUECOUNT',
    'M16A_BR_AVG_VELOCITY',
    'M16HUB.QUE.ALL.CURRENTQCREATED',
    'M16HUB.QUE.ALL.MESCURRENTQCNT'
]

def reorder_csv(input_path, output_path):
    # CSV 읽기
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(input_path, encoding='cp949')
        except:
            df = pd.read_csv(input_path, encoding='euc-kr')
    
    # 존재하는 컬럼만 필터링
    existing_cols = [c for c in COLUMN_ORDER if c in df.columns]
    
    # 누락된 컬럼 확인
    missing = [c for c in COLUMN_ORDER if c not in df.columns]
    if missing:
        print(f"⚠️ 누락된 컬럼: {missing}")
    
    # 순서대로 재배열
    df_ordered = df[existing_cols]
    
    # 저장
    df_ordered.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 저장 완료: {output_path}")
    print(f"   컬럼 수: {len(existing_cols)}")
    
    return df_ordered

# 사용 예시
# df = reorder_csv('input.csv', 'output_ordered.csv')
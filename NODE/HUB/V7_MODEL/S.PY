FEATURE_COLS = {
    # ========== V7 기존 (17개) ==========
    'storage': ['M16A_3F_STORAGE_UTIL'],
    
    'fs_storage': [
        'CD_M163FSTORAGEUSE',
        'CD_M163FSTORAGETOTAL',
        'CD_M163FSTORAGEUTIL'
    ],
    
    'hub': ['HUBROOMTOTAL'],
    
    'cmd': [
        'M16A_3F_CMD',
        'M16A_6F_TO_HUB_CMD'
    ],
    
    'inflow': [
        'M16A_6F_TO_HUB_JOB',
        'M16A_2F_TO_HUB_JOB2',
        'M14A_3F_TO_HUB_JOB2'
    ],
    
    'outflow': [
        'M16A_3F_TO_M16A_6F_JOB',
        'M16A_3F_TO_M16A_2F_JOB',
        'M16A_3F_TO_M14A_3F_JOB'
    ],
    
    'maxcapa': [
        'M16A_6F_LFT_MAXCAPA',
        'M16A_2F_LFT_MAXCAPA'
    ],
    
    # V7 QUE (2개)
    'que': [
        'M16HUB.QUE.ALL.CURRENTQCNT',
        'M16HUB.QUE.TIME.AVGTOTALTIME1MIN'
    ],
    
    # ========== V8 신규 추가 (16개) ==========
    
    # 🆕 TARGET 보완 (1개) - 매우 중요!
    'target_alt': ['CURRENT_M16A_3F_JOB'],
    
    # 🆕 M16HUB QUE 추가 (5개)
    'm16hub_que': [
        'M16HUB.QUE.ALL.CURRENTQCOMPLETED',    # HUB 반송완료수 (처리량)
        'M16HUB.QUE.ALL.FABTRANSJOBCNT',       # FAB간 반송 JOB수 (유입량)
        'M16HUB.QUE.TIME.AVGTOTALTIME',        # 평균 반송시간 전체 (병목)
        'M16HUB.QUE.OHT.CURRENTOHTQCNT',       # OHT 반송 Q (OHT부하)
        'M16HUB.QUE.OHT.OHTUTIL'               # OHT 사용율 (포화지표)
    ],
    
    # 🆕 M16A QUE 추가 (7개) - 핵심!
    'm16a_que': [
        'M16A.QUE.ALL.CURRENTQCOMPLETED',      # M16A 반송완료수 ⭐⭐⭐⭐⭐
        'M16A.QUE.ALL.CURRENTQCREATED',        # M16A 반송생성수 ⭐⭐⭐⭐⭐
        'M16A.QUE.OHT.CURRENTOHTQCNT',         # M16A OHT Q
        'M16A.QUE.OHT.OHTUTIL',                # M16A OHT 사용율
        'M16A.QUE.LOAD.AVGLOADTIME1MIN',       # 실시간 Load시간 ⭐⭐⭐⭐⭐
        'M16A.QUE.ALL.TRANSPORT4MINOVERCNT',   # 4분이상 반송건수 (선행지표) ⭐⭐⭐⭐⭐
        'M16A.QUE.ABN.QUETIMEDELAY'            # 20분이상 지연 (비상신호) ⭐⭐⭐⭐⭐
    ],
    
    # 🆕 Velocity (2개)
    'velocity': [
        'M16A_BR_AVG_VELOCITY',    # M16A 평균 속도
        'M14A_A_AVG_VELOCITY'      # M14A 평균 속도
    ],
    
    # ========== 선택 사항 (10개) ==========
    
    # BR_CNT Top 10 (선택적 추가 가능)
    'br_cnt_top10': [
        'M16A_BR_002_CNT',    # 평균 14.00 - Top1
        'M16A_BR_003_CNT',    # 평균 12.50 - Top2
        'M16A_BR_001_CNT',    # 평균 10.11 - Top3
        'M16A_BR_036_CNT',    # 평균 7.42 - Top4
        'M16A_BR_029_CNT',    # 평균 7.36 - Top5
        'M16A_BR_030_CNT',    # 평균 7.05 - Top6
        'M16A_BR_034_CNT',    # 평균 7.04 - Top7
        'M16A_BR_007_CNT',    # 평균 7.00 - Top8
        'M16A_BR_032_CNT',    # 평균 6.67 - Top9
        'M16A_BR_015_CNT'     # 평균 6.53 - Top10
    ]
}

TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
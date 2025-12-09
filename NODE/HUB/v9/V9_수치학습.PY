import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

def train_v9_regression():
    """
    🎯 V9-수치형 변화량 예측
    - V8 기존 14개 + 신규 161개 = 총 175개 컬럼
    - 타겟: [MIN 변화량, MAX 변화량, AVG 변화량]
    - 50+ 가중치 적용
    """
    print("="*80)
    print("🎯 V9-수치형 변화량 예측: 175개 컬럼 (기존14 + 신규161)")
    print("="*80)
    
    # ========== V8 기존 컬럼 (14개) ==========
    FEATURE_COLS_V8 = {
        'storage': ['M16A_3F_STORAGE_UTIL'],
        'fs_storage': ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL'],
        'hub': ['HUBROOMTOTAL'],
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA'],
        'que': ['M16HUB.QUE.ALL.CURRENTQCNT', 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN'],
        'target_alt': ['CURRENT_M16A_3F_JOB'],
        'm16hub_que': [
            'M16HUB.QUE.ALL.CURRENTQCOMPLETED',
            'M16HUB.QUE.ALL.FABTRANSJOBCNT',
            'M16HUB.QUE.TIME.AVGTOTALTIME',
            'M16HUB.QUE.OHT.CURRENTOHTQCNT',
            'M16HUB.QUE.OHT.OHTUTIL'
        ],
        'm16a_que': [
            'M16A.QUE.ALL.CURRENTQCOMPLETED',
            'M16A.QUE.ALL.CURRENTQCREATED',
            'M16A.QUE.OHT.CURRENTOHTQCNT',
            'M16A.QUE.OHT.OHTUTIL',
            'M16A.QUE.LOAD.AVGLOADTIME1MIN',
            'M16A.QUE.ALL.TRANSPORT4MINOVERCNT',
            'M16A.QUE.ABN.QUETIMEDELAY'
        ]
    }
    
    # ========== V9 신규 컬럼 (161개) ==========
    FEATURE_COLS_V9_NEW = {
        # M16HUB 신규 (10개)
        'm16hub_new': [
            'M16HUB.QUE.M16TOM14B.CURRENTQCREATED',
            'M16HUB.QUE.M14BTOM16.CURRENTQCREATED',
            'M16HUB.QUE.M14TOM16.CURRENTQCREATED',
            'M16HUB.QUE.M16TOM14.CURRENTQCREATED',
            'M16HUB.QUE.M14TOM16.MESCURRENTQCNT',
            'M16HUB.QUE.M16TOM14.MESCURRENTQCNT',
            'M16HUB.QUE.ALL.MESCURRENTQCNT',
            'M16HUB.QUE.M16TOM14A.MESCURRENTQCNT',
            'M16HUB.QUE.M16TOM14B.MESCURRENTQCNT',
            'M16HUB.QUE.ALL.M16HUBTOM14MANUAL_CURRENTQCNT'
        ],
        
        # M16A 신규 (19개)
        'm16a_new': [
            'M16A.QUE.LFT.ALLTOLFTCURRENTQCNT',
            'M16A.QUE.CNV.ALLTONORTHCNVCURRENTQCNT',
            'M16A.QUE.CNV.ALLTOSOUTHCNVCURRENTQCNT',
            'M16A.QUE.LFT.LFTTOALLCURRENTQCNT',
            'M16A.QUE.LOAD.AVGFOUPLOADTIME',
            'M16A.QUE.LOAD.AVGLOADTIME',
            'M16A.QUE.LOAD.AVGRETICLELOADTIME',
            'M16A.QUE.LOAD.CURRENTLOADQCNT',
            'M16A.QUE.LOAD.CURRENTRETICLELOADQCNT',
            'M16A.QUE.OHT.CURRENTRETICLEOHTQCNT',
            'M16A.QUE.CNV.M16ATOM14ACURRNETQCNT',
            'M16A.QUE.CNV.M16ATOM14BCURRNETQCNT',
            'M16A.QUE.CNV.NORTHCNVTOALLCURRENTQCNT',
            'M16A.QUE.CNV.SOUTHCNVTOALLCURRENTQCNT',
            'M16A.QUE.ALL.CURRENTQCNT',
            'M16A.QUE.ALL.CURRENTRETICLEQCNT',
            'M16A.QUE.ALL.TRANSPORT4MINOVERRATIO',
            'M16A.QUE.ALL.TRANSPORT4MINOVERTIMEAVG',
            'M16A.QUE.OHT.RTCOHTUTIL'
        ],
        
        # M14 (45개)
        'm14': [
            'M14.QUE.OHT.CURRENTRETICLEOHTQCNT',
            'M14.QUE.SENDFAB.VERTICALQUEUECOUNT',
            'M14.QUE.SENDFAB.CURRENTSENDFABCOMPLETED',
            'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.LOAD.CURRENTLOADQCNT',
            'M14.QUE.ALL.CURRENTQCNT',
            'M14.QUE.ALL.CURRENTQCOMPLETED',
            'M14.QUE.ALL.CURRENTQCREATED',
            'M14.QUE.LOAD.AVGLOADTIME',
            'M14.QUE.OHT.RTCOHTUTIL',
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT',
            'M14.QUE.ALL.TRANSPORT4MINOVERTIMEAVG',
            'M14.QUE.ALL.TRANSPORT4MINOVERRATIO',
            'M14.QUE.CNV.M14ATOM16ACURRNETQCNT',
            'M14.QUE.CNV.ALLTONORTHCNVCURRENTQCNT',
            'M14.QUE.CNV.ALLTOSOUTHCNVCURRENTQCNT',
            'M14.QUE.LOAD.AVGRETICLELOADTIME',
            'M14.QUE.LOAD.AVGLOADTIME1MIN',
            'M14.QUE.OHT.CURRENTOHTQCNT',
            'M14.QUE.LOAD.AVGFOUPLOADTIME',
            'M14.QUE.ALL.CURRENTRETICLEQCNT',
            'M14.QUE.LOAD.CURRENTRETICLELOADQCNT',
            'M14.QUE.STK.MANUALOUTCOUNT',
            'M14.QUE.CNV.NORTHCNVTOALLCURRENTQCNT',
            'M14.QUE.CNV.SOUTHCNVTOALLCURRENTQCNT',
            'M14.QUE.ALL.TOTALCNVCURRENTQCNT',
            'M14.QUE.ALL.M14ATOM10ACURRENTQCNT',
            'M14.QUE.ALL.M14ATOM10CCURRENTQCNT',
            'M14.QUE.CNV.NORTHM14TOCNVTIME',
            'M14.QUE.CNV.SOUTHM14TOCNVTIME',
            'M14.QUE.SFAB.SENDTOR3',
            'M14.QUE.SFAB.SENDTOM16',
            'M14.QUE.SFAB.SENDTOM10',
            'M14.QUE.SFAB.SENDQUEUETOTAL',
            'M14.QUE.SFAB.RECEIVETOR3',
            'M14.QUE.SFAB.RECEIVETOM16',
            'M14.QUE.SFAB.RECEIVETOM10',
            'M14.QUE.SFAB.RECEIVEQUEUETOTAL',
            'M14.QUE.SFAB.RETURNTOR3',
            'M14.QUE.SFAB.RETURNTOM16',
            'M14.QUE.SFAB.RETURNTOM10',
            'M14.QUE.SFAB.RETURNQUEUETOTAL',
            'M14.QUE.SFAB.COMPLETETOR3',
            'M14.QUE.SFAB.COMPLETETOM16',
            'M14.QUE.SFAB.COMPLETETOM10'
        ],
        
        # M14B (25개)
        'm14b': [
            'M14B.QUE.LOAD.CURRENTRETICLELOADQCNT',
            'M14B.QUE.SENDFAB.VERTICALQUEUECOUNT',
            'M14B.QUE.SENDFAB.CURRENTSENDFABCOMPLETED',
            'M14B.QUE.OHT.RTCOHTUTIL',
            'M14B.QUE.LFT.ALLTOLFTCURRENTQCNT',
            'M14B.QUE.LFT.LFTTOALLCURRENTQCNT',
            'M14B.QUE.LOAD.CURRENTLOADQCNT',
            'M14B.QUE.ALL.CURRENTQCNT',
            'M14B.QUE.ALL.CURRENTQCOMPLETED',
            'M14B.QUE.ALL.CURRENTQCREATED',
            'M14B.QUE.LOAD.AVGLOADTIME',
            'M14B.QUE.OHT.OHTUTIL',
            'M14B.QUE.ABN.QUETIMEDELAY',
            'M14B.QUE.OHT.CURRENTRETICLEOHTQCNT',
            'M14B.QUE.LOAD.AVGRETICLELOADTIME',
            'M14B.QUE.LOAD.AVGLOADTIME1MIN',
            'M14B.QUE.OHT.CURRENTOHTQCNT',
            'M14B.QUE.LOAD.AVGFOUPLOADTIME',
            'M14B.QUE.ALL.CURRENTRETICLEQCNT',
            'M14B.QUE.STK.MANUALOUTCOUNT',
            'M14B.QUE.LFT.M14BTOM16ACURRNETQCNT',
            'M14B.QUE.ALL.M14BTOM10ACURRENTQCNT',
            'M14B.QUE.ALL.M14BTOM16CURRENTQCNT',
            'M14B.QUE.ALL.FABOUTCURRENTQCNT',
            'M14B.QUE.ALL.TRANSPORT4MINOVERCNT'
        ],
        
        # M16 SFAB (14개)
        'm16_sfab': [
            'M16.QUE.SFAB.SENDTOR3',
            'M16.QUE.SFAB.SENDTOM14',
            'M16.QUE.SFAB.SENDTOM10',
            'M16.QUE.SFAB.SENDQUEUETOTAL',
            'M16.QUE.SFAB.RECEIVETOR3',
            'M16.QUE.SFAB.RECEIVETOM14',
            'M16.QUE.SFAB.RECEIVEQUEUETOTAL',
            'M16.QUE.SFAB.RETURNTOR3',
            'M16.QUE.SFAB.RETURNTOM14',
            'M16.QUE.SFAB.RETURNQUEUETOTAL',
            'M16.QUE.SFAB.COMPLETETOR3',
            'M16.QUE.SFAB.COMPLETETOM14',
            'M16.QUE.SFAB.COMPLETETOM10',
            'M16.QUE.SFAB.COMPLETEQUEUETOTAL'
        ],
        
        # M16B (18개)
        'm16b': [
            'M16B.QUE.OHT.CURRENTOHTQCNT',
            'M16B.QUE.OHT.CURRENTRETICLEOHTQCNT',
            'M16B.QUE.OHT.OHTUTIL',
            'M16B.QUE.ALL.CURRENTQCREATED',
            'M16B.QUE.ALL.CURRENTRETICLEQCNT',
            'M16B.QUE.ALL.TRANSPORT4MINOVERCNT',
            'M16B.QUE.ALL.TRANSPORT4MINOVERRATIO',
            'M16B.QUE.ALL.TRANSPORT4MINOVERTIMEAVG',
            'M16B.QUE.OHT.RTCOHTUTIL',
            'M16B.QUE.ABN.QUETIMEDELAY',
            'M16B.QUE.ALL.CURRENTQCNT',
            'M16B.QUE.ALL.CURRENTQCOMPLETED',
            'M16B.QUE.LOAD.AVGFOUPLOADTIME',
            'M16B.QUE.LOAD.AVGLOADTIME',
            'M16B.QUE.LOAD.AVGLOADTIME1MIN',
            'M16B.QUE.LOAD.AVGRETICLELOADTIME',
            'M16B.QUE.LOAD.CURRENTLOADQCNT',
            'M16B.QUE.LOAD.CURRENTRETICLELOADQCNT'
        ],
        
        # M16E (20개)
        'm16e': [
            'M16E.QUE.ALL.M16ETOM14BCURRENTQCNT',
            'M16E.QUE.ALL.CURRENTQCNT',
            'M16E.QUE.ALL.CURRENTQCOMPLETED',
            'M16E.QUE.ALL.CURRENTQCREATED',
            'M16E.QUE.ALL.CURRENTRETICLEQCNT',
            'M16E.QUE.ALL.TRANSPORT4MINOVERCNT',
            'M16E.QUE.ALL.TRANSPORT4MINOVERRATIO',
            'M16E.QUE.ALL.TRANSPORT4MINOVERTIMEAVG',
            'M16E.QUE.LOAD.AVGFOUPLOADTIME',
            'M16E.QUE.LOAD.AVGLOADTIME',
            'M16E.QUE.LOAD.AVGLOADTIME1MIN',
            'M16E.QUE.LOAD.AVGRETICLELOADTIME',
            'M16E.QUE.LOAD.CURRENTLOADQCNT',
            'M16E.QUE.OHT.CURRENTOHTQCNT',
            'M16E.QUE.OHT.CURRENTRETICLEOHTQCNT',
            'M16E.QUE.OHT.OHTUTIL',
            'M16E.QUE.OHT.RTCOHTUTIL',
            'M16E.QUE.ALL.M14ATOM16ECURRENTQCNT',
            'M16E.QUE.ALL.M14BTOM16ECURRENTQCNT',
            'M16E.QUE.ALL.M16ETOM16ACURRENTQCNT'
        ]
    }
    
    # 전체 컬럼 합치기
    FEATURE_COLS = {**FEATURE_COLS_V8, **FEATURE_COLS_V9_NEW}
    
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    def create_features_v9(df, available_cols, start_idx=30):
        """V9 Feature 생성 - 175개 컬럼"""
        features_list = []
        labels = []
        indices = []
        
        total_samples = len(df) - 10 - start_idx
        
        for idx, i in enumerate(range(start_idx, len(df) - 10)):
            if idx % 1000 == 0:
                print(f"  Feature 생성: {idx}/{total_samples} ({idx/total_samples*100:.1f}%)")
            
            seq_target = df[TARGET_COL].iloc[i-30:i].values
            current_value = seq_target[-1]
            
            future_10min = df[TARGET_COL].iloc[i:i+10].values
            future_min = np.min(future_10min)
            future_max = np.max(future_10min)
            future_avg = np.mean(future_10min)
            
            change_min = future_min - current_value
            change_max = future_max - current_value
            change_avg = future_avg - current_value
            
            # 타겟 기본 통계
            features = {
                'target_mean': np.mean(seq_target),
                'target_std': np.std(seq_target),
                'target_max': np.max(seq_target),
                'target_min': np.min(seq_target),
                'target_last_value': seq_target[-1],
                'target_last_5_mean': np.mean(seq_target[-5:]),
                'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
                'target_acceleration': (seq_target[-5:].mean() - seq_target[-10:-5].mean()) / 5,
                'target_is_rising': 1 if seq_target[-1] > seq_target[-5] else 0,
                'target_rapid_rise': 1 if (seq_target[-1] - seq_target[-5] > 10) else 0,
                'target_last_10_mean': np.mean(seq_target[-10:])
            }
            
            # 각 컬럼 그룹 Feature
            for group_name, cols in FEATURE_COLS.items():
                for col in cols:
                    if col not in available_cols:
                        continue
                    
                    col_seq = df[col].iloc[i-30:i].values
                    
                    # 그룹별 Feature 생성 방식
                    if group_name == 'maxcapa':
                        features[f'{col}_last_value'] = col_seq[-1]
                    
                    elif group_name in ['inflow', 'outflow']:
                        features[f'{col}_mean'] = np.mean(col_seq)
                        features[f'{col}_last_value'] = col_seq[-1]
                        features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
                    
                    else:  # 대부분의 컬럼 (V8 + V9 신규)
                        features[f'{col}_mean'] = np.mean(col_seq)
                        features[f'{col}_std'] = np.std(col_seq)
                        features[f'{col}_max'] = np.max(col_seq)
                        features[f'{col}_min'] = np.min(col_seq)
                        features[f'{col}_last_value'] = col_seq[-1]
                        features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
                        features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
            
            # FS Storage 특수 Feature (V8)
            if all(col in available_cols for col in ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL']):
                storage_use = df['CD_M163FSTORAGEUSE'].iloc[i-30:i].values
                storage_total = df['CD_M163FSTORAGETOTAL'].iloc[i-30:i].values
                storage_util = df['CD_M163FSTORAGEUTIL'].iloc[i-30:i].values
                
                features['storage_use_rate'] = (storage_use[-1] - storage_use[0]) / 30
                features['storage_remaining'] = storage_total[-1] - storage_use[-1]
                features['storage_util_last'] = storage_util[-1]
                features['storage_util_high'] = 1 if storage_util[-1] >= 7 else 0
                features['storage_util_critical'] = 1 if storage_util[-1] >= 10 else 0
            
            # HUBROOMTOTAL 특수 Feature (V8)
            if 'HUBROOMTOTAL' in available_cols:
                hub_seq = df['HUBROOMTOTAL'].iloc[i-30:i].values
                hub_last = hub_seq[-1]
                
                features['hub_critical'] = 1 if hub_last < 590 else 0
                features['hub_high'] = 1 if hub_last < 610 else 0
                features['hub_warning'] = 1 if hub_last < 620 else 0
                features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30
                
                if 'CD_M163FSTORAGEUTIL' in available_cols:
                    storage_util_last = df['CD_M163FSTORAGEUTIL'].iloc[i-1]
                    features['hub_storage_risk'] = 1 if (hub_last < 610 and storage_util_last >= 7) else 0
            
            # 복합 Feature (V8)
            inflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS_V8.get('inflow', []) if col in available_cols)
            outflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS_V8.get('outflow', []) if col in available_cols)
            features['net_flow'] = inflow_sum - outflow_sum
            
            cmd_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS_V8.get('cmd', []) if col in available_cols)
            features['total_cmd'] = cmd_sum
            features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
            features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
            
            if 'HUBROOMTOTAL' in available_cols:
                hub_last = df['HUBROOMTOTAL'].iloc[i-1]
                features['hub_cmd_bottleneck'] = 1 if (hub_last < 610 and cmd_sum < 220) else 0
            
            if 'M16A_3F_STORAGE_UTIL' in available_cols:
                storage_util = df['M16A_3F_STORAGE_UTIL'].iloc[i-1]
                features['storage_util_critical'] = 1 if storage_util >= 205 else 0
                features['storage_util_high_risk'] = 1 if storage_util >= 207 else 0
            
            features['surge_risk_score'] = (
                features.get('hub_high', 0) * 3 +
                features.get('storage_util_critical', 0) * 2 +
                features.get('total_cmd_low', 0) * 1 +
                features.get('storage_util_high', 0) * 1
            )
            
            features['surge_imminent'] = 1 if (
                seq_target[-1] > 280 and
                features.get('target_acceleration', 0) > 0.5 and
                features.get('hub_high', 0) == 1
            ) else 0
            
            # V8 Boolean Features
            if 'M16HUB.QUE.ALL.CURRENTQCNT' in available_cols:
                currentq = df['M16HUB.QUE.ALL.CURRENTQCNT'].iloc[i-1]
                features['currentq_high'] = 1 if currentq >= 1200 else 0
                features['currentq_critical'] = 1 if currentq >= 1400 else 0
            
            if 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN' in available_cols:
                avgtime = df['M16HUB.QUE.TIME.AVGTOTALTIME1MIN'].iloc[i-1]
                features['avgtime1min_high'] = 1 if avgtime >= 4.0 else 0
                features['avgtime1min_critical'] = 1 if avgtime >= 4.5 else 0
            
            if 'M16HUB.QUE.OHT.OHTUTIL' in available_cols:
                ohtutil = df['M16HUB.QUE.OHT.OHTUTIL'].iloc[i-1]
                features['m16hub_ohtutil_high'] = 1 if ohtutil >= 85.0 else 0
                features['m16hub_ohtutil_critical'] = 1 if ohtutil >= 90.0 else 0
            
            if 'M16A.QUE.OHT.OHTUTIL' in available_cols:
                ohtutil = df['M16A.QUE.OHT.OHTUTIL'].iloc[i-1]
                features['m16a_ohtutil_high'] = 1 if ohtutil >= 85.0 else 0
                features['m16a_ohtutil_critical'] = 1 if ohtutil >= 90.0 else 0
            
            # V9 신규 Boolean Features
            # M14 OHT
            if 'M14.QUE.OHT.OHTUTIL' in available_cols:
                ohtutil = df['M14.QUE.OHT.OHTUTIL'].iloc[i-1]
                features['m14_ohtutil_high'] = 1 if ohtutil >= 85.0 else 0
                features['m14_ohtutil_critical'] = 1 if ohtutil >= 90.0 else 0
            
            # M14B OHT
            if 'M14B.QUE.OHT.OHTUTIL' in available_cols:
                ohtutil = df['M14B.QUE.OHT.OHTUTIL'].iloc[i-1]
                features['m14b_ohtutil_high'] = 1 if ohtutil >= 85.0 else 0
                features['m14b_ohtutil_critical'] = 1 if ohtutil >= 90.0 else 0
            
            # M16B OHT
            if 'M16B.QUE.OHT.OHTUTIL' in available_cols:
                ohtutil = df['M16B.QUE.OHT.OHTUTIL'].iloc[i-1]
                features['m16b_ohtutil_high'] = 1 if ohtutil >= 85.0 else 0
                features['m16b_ohtutil_critical'] = 1 if ohtutil >= 90.0 else 0
            
            # M16E OHT
            if 'M16E.QUE.OHT.OHTUTIL' in available_cols:
                ohtutil = df['M16E.QUE.OHT.OHTUTIL'].iloc[i-1]
                features['m16e_ohtutil_high'] = 1 if ohtutil >= 85.0 else 0
                features['m16e_ohtutil_critical'] = 1 if ohtutil >= 90.0 else 0
            
            # FAB간 흐름 Boolean
            if 'M16HUB.QUE.M16TOM14.CURRENTQCREATED' in available_cols:
                m16tom14 = df['M16HUB.QUE.M16TOM14.CURRENTQCREATED'].iloc[i-1]
                features['m16tom14_high'] = 1 if m16tom14 >= 50 else 0
            
            if 'M16HUB.QUE.M14TOM16.CURRENTQCREATED' in available_cols:
                m14tom16 = df['M16HUB.QUE.M14TOM16.CURRENTQCREATED'].iloc[i-1]
                features['m14tom16_high'] = 1 if m14tom16 >= 50 else 0
            
            features_list.append(features)
            labels.append([change_min, change_max, change_avg])
            indices.append(i)
        
        return pd.DataFrame(features_list), np.array(labels), indices
    
    # ========== 학습 시작 ==========
    print("\n[STEP 1] 데이터 로드")
    print("-"*40)
    
    try:
        df_train = pd.read_csv('20250904_TO_20251020.csv', on_bad_lines='skip', encoding='utf-8', low_memory=False)
    except:
        try:
            df_train = pd.read_csv('20250904_TO_20251020.csv', on_bad_lines='skip', encoding='cp949', low_memory=False)
        except:
            df_train = pd.read_csv('20250904_TO_20251020.csv', on_bad_lines='skip', encoding='euc-kr', low_memory=False)
    
    df_train[TARGET_COL] = pd.to_numeric(df_train[TARGET_COL], errors='coerce')
    df_train = df_train.dropna(subset=[TARGET_COL])
    
    print(f"학습 데이터: {len(df_train)}개 행")
    
    # 컬럼 존재 확인
    available_cols = set(df_train.columns)
    
    print("\n📋 V9 컬럼 확인:")
    v8_count = sum(1 for cols in FEATURE_COLS_V8.values() for col in cols if col in available_cols)
    v9_new_count = sum(1 for cols in FEATURE_COLS_V9_NEW.values() for col in cols if col in available_cols)
    total_expected = sum(len(cols) for cols in FEATURE_COLS.values())
    
    print(f"  V8 기존: {v8_count}개")
    print(f"  V9 신규: {v9_new_count}개")
    print(f"  총 사용 가능: {v8_count + v9_new_count}/{total_expected}개")
    
    # Feature 생성
    print("\n[STEP 2] Feature 생성")
    print("-"*40)
    X_train, y_train, _ = create_features_v9(df_train, available_cols)
    
    print(f"\n✅ Feature 생성 완료:")
    print(f"  - Feature: {len(X_train.columns)}개")
    print(f"  - 샘플: {len(X_train)}개")
    
    # 가중치 계산
    print("\n🔥 가중치 계산 (50+ 샘플 강조):")
    large_changes = np.abs(y_train[:, 1]) >= 50
    sample_weights = np.where(large_changes, 3.0, 1.0)
    
    large_count = large_changes.sum()
    normal_count = len(large_changes) - large_count
    
    print(f"  - 일반 샘플 (가중치 1.0): {normal_count}개")
    print(f"  - 큰 변화 (가중치 3.0): {large_count}개 ({large_count/len(large_changes)*100:.1f}%)")
    
    # 학습/검증 분할
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train, y_train, sample_weights, test_size=0.2, random_state=42
    )
    
    # GPU/CPU 감지
    print("\n🔍 학습 환경 감지...")
    use_gpu = False
    
    try:
        test_model = xgb.XGBRegressor(
            n_estimators=5, max_depth=3, random_state=42,
            tree_method='gpu_hist', gpu_id=0
        )
        test_model.fit(X_tr[:100], y_tr[:100, 0], verbose=False)
        use_gpu = True
        print("  ✅ GPU 모드\n")
    except:
        print("  ⚠️ CPU 모드\n")
        use_gpu = False
    
    print("[STEP 3] 모델 학습 (3개 개별)")
    print("="*60)
    
    models = []
    
    for idx, name in enumerate(['MIN', 'MAX', 'AVG']):
        print(f"\n[{name}] 학습 중...")
        
        if use_gpu:
            individual_model = xgb.XGBRegressor(
                n_estimators=250,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=2,
                gamma=0.05,
                reg_alpha=0.05,
                reg_lambda=0.8,
                random_state=42,
                tree_method='gpu_hist',
                gpu_id=0,
                predictor='gpu_predictor'
            )
        else:
            individual_model = xgb.XGBRegressor(
                n_estimators=250,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=2,
                gamma=0.05,
                reg_alpha=0.05,
                reg_lambda=0.8,
                random_state=42,
                tree_method='hist',
                n_jobs=-1
            )
        
        individual_model.fit(X_tr, y_tr[:, idx], sample_weight=w_tr, verbose=False)
        models.append(individual_model)
        print(f"✓ {name} 학습 완료")
    
    print("="*60)
    print("✅ 학습 완료!")
    
    # 평가
    y_val_pred = np.column_stack([
        models[0].predict(X_val),
        models[1].predict(X_val),
        models[2].predict(X_val)
    ])
    
    print(f"\n📊 학습 성능:")
    for idx, name in enumerate(['MIN', 'MAX', 'AVG']):
        mae = mean_absolute_error(y_val[:, idx], y_val_pred[:, idx])
        rmse = np.sqrt(mean_squared_error(y_val[:, idx], y_val_pred[:, idx]))
        r2 = r2_score(y_val[:, idx], y_val_pred[:, idx])
        print(f"  {name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # 큰 변화 성능
    print(f"\n🎯 큰 변화(±50) 예측 성능:")
    large_idx = np.abs(y_val[:, 1]) >= 50
    if large_idx.sum() > 0:
        mae_large = mean_absolute_error(y_val[large_idx, 1], y_val_pred[large_idx, 1])
        print(f"  - 큰 변화 샘플: {large_idx.sum()}개")
        print(f"  - MAX MAE (큰 변화만): {mae_large:.4f}")
    
    # 모델 저장
    model_dict = {
        'models': models,
        'feature_names': X_train.columns.tolist(),
        'version': 'V9',
        'total_columns': v8_count + v9_new_count
    }
    
    model_filename = 'xgboost_수치형_V9.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"\n✅ 모델 저장: {model_filename}")
    
    # Feature 중요도 Top 30
    print("\n🔥 Feature 중요도 Top 30 (MAX 모델 기준):")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': models[1].feature_importances_
    }).sort_values('importance', ascending=False).head(30)
    
    for rank, (idx, row) in enumerate(feature_importance.iterrows(), 1):
        print(f"  {rank:2d}. {row['feature']}: {row['importance']:.4f}")
    
    print("\n✅ V9-수치형 학습 완료!")

if __name__ == '__main__':
    print("🚀 V9-수치형 변화량 예측 모델 학습 (175개 컬럼)\n")
    train_v9_regression()
    print(f"\n🎉 완료!")
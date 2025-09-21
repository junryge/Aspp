# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 20:08:54 2025

@author: ggg3g
"""

# -*- coding: utf-8 -*-
"""
3구간 분류 모델 평가 시스템
학습된 모델로 새로운 CSV 파일을 평가하고 상세한 예측 결과 출력
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (15, 10)

print("="*80)
print("3구간 분류 모델 평가 시스템")
print("="*80)
print("학습된 모델로 새로운 CSV 파일 평가 및 예측 결과 출력")
print("="*80)

# 1. 설정
model_dir = 'regression_classification_models_280to10'
output_dir = 'prediction_results'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 구간 분류 함수 (학습 시와 동일)
def assign_level(totalcnt_value):
    """TOTALCNT 값을 3구간으로 분류"""
    if totalcnt_value < 1400:
        return 0  # 정상 (900-1399)
    elif totalcnt_value < 1700:
        return 1  # 확인 (1400-1699)
    else:
        return 2  # 위험 (1700+)

def detect_anomaly_signal(totalcnt_value):
    """1651-1682 이상신호 감지"""
    return 1 if 1651 <= totalcnt_value <= 1682 else 0

def get_level_name(level):
    """구간 번호를 이름으로 변환"""
    level_names = {0: '정상', 1: '확인', 2: '위험'}
    return level_names.get(level, '알수없음')

# 3. 시퀀스 생성 함수 (평가용)
def create_sequences_for_evaluation(data, seq_length=280, pred_horizon=10):
    """
    평가용 280분 시퀀스 생성 (학습 시와 동일한 특징 추출)
    """
    print(f"\n평가용 시퀀스 생성 중... (280분 → 10분 후)")
    
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 'M16M14A', 'TOTALCNT']
    
    # 파생 변수 생성
    data = data.copy()
    data['ratio_M14B_M14A'] = data['M14AM14B'] / (data['M14AM10A'] + 1)
    data['ratio_M14B_M16'] = data['M14AM14B'] / (data['M14AM16'] + 1)
    data['totalcnt_change'] = data['TOTALCNT'].diff().fillna(0)
    data['totalcnt_pct_change'] = data['TOTALCNT'].pct_change().fillna(0)
    
    X_list = []
    y_reg_list = []
    y_cls_list = []
    y_anomaly_list = []
    time_info_list = []  # 시간 정보 저장
    
    n_sequences = len(data) - seq_length - pred_horizon + 1
    print(f"✓ 생성 가능한 시퀀스: {n_sequences:,}개")
    
    for i in range(n_sequences):
        if i % 1000 == 0:
            print(f"  진행률: {i/n_sequences*100:.1f}%", end='\r')
        
        start_idx = i
        end_idx = i + seq_length
        seq_data = data.iloc[start_idx:end_idx]
        
        features = []
        
        # 각 컬럼별 특징 추출 (학습 시와 완전히 동일)
        for col in feature_cols:
            values = seq_data[col].values
            
            # 기본 통계
            features.extend([
                np.mean(values),                    # 평균
                np.std(values),                     # 표준편차
                np.min(values),                     # 최소값
                np.max(values),                     # 최대값
                np.percentile(values, 25),          # Q1
                np.percentile(values, 50),          # 중간값
                np.percentile(values, 75),          # Q3
                values[-1],                         # 현재값 (280분째)
                values[-1] - values[0],             # 전체 변화량
                np.mean(values[-60:]),              # 최근 1시간 평균
                np.max(values[-60:]),               # 최근 1시간 최대
                np.mean(values[-30:]),              # 최근 30분 평균
                np.max(values[-30:]),               # 최근 30분 최대
            ])
            
            # TOTALCNT 특별 처리
            if col == 'TOTALCNT':
                # 기존 위험 구간 관련
                features.append(np.sum((values >= 1650) & (values < 1700)))  # 경고 구간
                features.append(np.sum(values >= 1700))                      # 위험 횟수
                features.append(np.max(values[-20:]))                        # 최근 20분 최대
                
                # 3구간 관련 특징
                features.append(np.sum(values < 1400))                       # 정상 구간 횟수
                features.append(np.sum((values >= 1400) & (values < 1700)))  # 확인 구간 횟수
                features.append(np.sum(values >= 1700))                      # 위험 구간 횟수
                
                # 이상신호 1651-1682 구간 특징
                features.append(np.sum((values >= 1651) & (values <= 1682))) # 이상신호 횟수
                features.append(np.max(values[(values >= 1651) & (values <= 1682)]) if np.any((values >= 1651) & (values <= 1682)) else 0)
                
                # 구간별 평균값
                normal_vals = values[values < 1400]
                check_vals = values[(values >= 1400) & (values < 1700)]
                danger_vals = values[values >= 1700]
                
                features.append(np.mean(normal_vals) if len(normal_vals) > 0 else 0)
                features.append(np.mean(check_vals) if len(check_vals) > 0 else 0)
                features.append(np.mean(danger_vals) if len(danger_vals) > 0 else 0)
                
                # 추세 분석
                x = np.arange(len(values))
                slope, intercept = np.polyfit(x, values, 1)
                features.append(slope)
                
                recent_slope = np.polyfit(np.arange(60), values[-60:], 1)[0]
                features.append(recent_slope)
        
        # 마지막 시점의 파생 변수들
        last_idx = end_idx - 1
        features.extend([
            data['ratio_M14B_M14A'].iloc[last_idx],
            data['ratio_M14B_M16'].iloc[last_idx],
            data['totalcnt_change'].iloc[last_idx],
            data['totalcnt_pct_change'].iloc[last_idx],
        ])
        
        # 타겟: 10분 후 TOTALCNT
        target_idx = end_idx + pred_horizon - 1
        future_totalcnt = data['TOTALCNT'].iloc[target_idx]
        
        # 시간 정보 저장 (인덱스 기반)
        time_info = {
            'sequence_start_idx': start_idx,
            'sequence_end_idx': end_idx - 1,  # 280분째 (현재 시점)
            'target_idx': target_idx,         # 290분째 (10분 후)
            'current_time_idx': end_idx - 1,  # 현재 시점
            'prediction_time_idx': target_idx  # 예측 시점
        }
        
        X_list.append(features)
        y_reg_list.append(future_totalcnt)
        y_cls_list.append(assign_level(future_totalcnt))
        y_anomaly_list.append(detect_anomaly_signal(future_totalcnt))
        time_info_list.append(time_info)
    
    print("\n✓ 시퀀스 생성 완료!")
    
    return np.array(X_list), np.array(y_reg_list), np.array(y_cls_list), np.array(y_anomaly_list), time_info_list

# 4. 모델 평가 및 예측 함수
def evaluate_and_predict_csv(csv_path):
    """
    CSV 파일로 모델 평가 및 상세 예측 결과 생성
    """
    
    print(f"\n[평가 데이터 로드]")
    print(f"파일: {csv_path}")
    
    # 1. CSV 로드
    df = pd.read_csv(csv_path)
    print(f"✓ 데이터 크기: {len(df):,}행")
    
    # 필수 컬럼 확인
    required_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                    'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 'M16M14A', 'TOTALCNT']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 누락된 컬럼: {missing_cols}")
        return None
    
    print(f"✓ 필수 컬럼 확인 완료")
    
    # 2. 시퀀스 생성
    X, y_reg, y_cls, y_anomaly, time_info = create_sequences_for_evaluation(df)
    print(f"\n✓ 평가 시퀀스: {X.shape[0]:,}개")
    
    # 3. 스케일러 및 모델 로드
    print(f"\n[모델 로드]")
    try:
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        print("✓ 스케일러 로드 완료")
        
        # 설정 로드
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        print("✓ 설정 로드 완료")
        
        # 최고 성능 모델들 로드
        best_reg_model_name = config['best_regression_models']['r2']
        best_cls_model_name = config['best_classification_model']
        
        reg_model = joblib.load(os.path.join(model_dir, f'{best_reg_model_name}_regression_model.pkl'))
        cls_model = joblib.load(os.path.join(model_dir, f'{best_cls_model_name}_classification_model.pkl'))
        
        print(f"✓ 최고 회귀 모델 로드: {best_reg_model_name}")
        print(f"✓ 최고 분류 모델 로드: {best_cls_model_name}")
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {str(e)}")
        print("먼저 모델을 학습시켜 주세요.")
        return None
    
    # 4. 예측 수행
    print(f"\n[예측 수행]")
    X_scaled = scaler.transform(X)
    
    # 회귀 예측
    y_reg_pred = reg_model.predict(X_scaled)
    
    # 분류 예측
    y_cls_pred = cls_model.predict(X_scaled)
    y_cls_prob = cls_model.predict_proba(X_scaled) if hasattr(cls_model, 'predict_proba') else None
    
    print("✓ 예측 완료")
    
    # 5. 상세 결과 생성
    print(f"\n[결과 생성]")
    
    results_data = []
    
    for i in range(len(X)):
        # 시간 정보
        time_info_item = time_info[i]
        current_idx = time_info_item['current_time_idx']
        target_idx = time_info_item['target_idx']
        
        # 현재 시점과 예측 시점의 실제 데이터
        current_totalcnt = df['TOTALCNT'].iloc[current_idx] if current_idx < len(df) else None
        
        # 예측 결과
        reg_pred = y_reg_pred[i]
        cls_pred = y_cls_pred[i]
        
        # 실제 값
        reg_actual = y_reg[i]
        cls_actual = y_cls[i]
        anomaly_actual = y_anomaly[i]
        
        # 오차 계산
        reg_error = reg_pred - reg_actual
        reg_error_pct = (reg_error / reg_actual) * 100 if reg_actual != 0 else 0
        cls_correct = cls_pred == cls_actual
        
        # 예측값 기반 이상신호 감지
        anomaly_pred = detect_anomaly_signal(reg_pred)
        
        # 신뢰도 (분류 확률에서 최대값)
        confidence = np.max(y_cls_prob[i]) if y_cls_prob is not None else 0.0
        
        result_row = {
            # 시간 정보
            '시퀀스번호': i + 1,
            '현재시점_인덱스': current_idx,
            '예측시점_인덱스': target_idx,
            '현재_TOTALCNT': current_totalcnt,
            
            # 실제 값
            '실제_TOTALCNT': reg_actual,
            '실제_구간': cls_actual,
            '실제_구간명': get_level_name(cls_actual),
            '실제_이상신호': anomaly_actual,
            
            # 예측 값
            '예측_TOTALCNT': round(reg_pred, 1),
            '예측_구간': cls_pred,
            '예측_구간명': get_level_name(cls_pred),
            '예측_이상신호': anomaly_pred,
            
            # 오차 및 성능
            '회귀_오차': round(reg_error, 1),
            '회귀_오차율(%)': round(reg_error_pct, 2),
            '구간_정확': cls_correct,
            '위험_실제': reg_actual >= 1700,
            '위험_예측': reg_pred >= 1700,
            '위험_정확': (reg_actual >= 1700) == (reg_pred >= 1700),
            
            # 신뢰도
            '신뢰도': round(confidence, 3),
            
            # 상태
            '상태': '정확' if cls_correct and abs(reg_error) < 50 else '부정확'
        }
        
        results_data.append(result_row)
    
    # 6. DataFrame 생성
    results_df = pd.DataFrame(results_data)
    
    # 7. 성능 지표 계산
    print(f"\n[성능 지표 계산]")
    
    # 회귀 성능
    reg_mae = mean_absolute_error(y_reg, y_reg_pred)
    reg_rmse = np.sqrt(mean_squared_error(y_reg, y_reg_pred))
    reg_r2 = r2_score(y_reg, y_reg_pred)
    
    # 분류 성능
    cls_accuracy = accuracy_score(y_cls, y_cls_pred)
    
    # 위험 감지 성능
    danger_actual = y_reg >= 1700
    danger_pred = y_reg_pred >= 1700
    danger_tp = np.sum(danger_actual & danger_pred)
    danger_fp = np.sum(~danger_actual & danger_pred)
    danger_fn = np.sum(danger_actual & ~danger_pred)
    danger_precision = danger_tp / (danger_tp + danger_fp) if (danger_tp + danger_fp) > 0 else 0
    danger_recall = danger_tp / (danger_tp + danger_fn) if (danger_tp + danger_fn) > 0 else 0
    danger_f1 = 2 * (danger_precision * danger_recall) / (danger_precision + danger_recall) if (danger_precision + danger_recall) > 0 else 0
    
    # 구간별 분포
    level_distribution = results_df.groupby(['실제_구간', '예측_구간']).size().unstack(fill_value=0)
    
    print(f"✓ 회귀 성능: MAE={reg_mae:.2f}, RMSE={reg_rmse:.2f}, R²={reg_r2:.4f}")
    print(f"✓ 분류 성능: 정확도={cls_accuracy:.3f}")
    print(f"✓ 위험 감지: Precision={danger_precision:.3f}, Recall={danger_recall:.3f}, F1={danger_f1:.3f}")
    
    # 8. 결과 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 상세 결과 CSV 저장
    detail_file = os.path.join(output_dir, f'prediction_details_{timestamp}.csv')
    results_df.to_csv(detail_file, index=False, encoding='utf-8-sig')
    print(f"✓ 상세 결과 저장: {detail_file}")
    
    # 요약 보고서 저장
    summary_data = {
        '평가일시': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '평가파일': csv_path,
        '총_시퀀스수': len(results_df),
        '회귀_MAE': reg_mae,
        '회귀_RMSE': reg_rmse,
        '회귀_R2': reg_r2,
        '분류_정확도': cls_accuracy,
        '위험감지_Precision': danger_precision,
        '위험감지_Recall': danger_recall,
        '위험감지_F1': danger_f1,
        '사용된_회귀모델': best_reg_model_name,
        '사용된_분류모델': best_cls_model_name
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_file = os.path.join(output_dir, f'prediction_summary_{timestamp}.csv')
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"✓ 요약 보고서 저장: {summary_file}")
    
    # 9. 화면 출력 (상위 20개)
    print(f"\n" + "="*100)
    print("예측 결과 (상위 20개)")
    print("="*100)
    
    display_cols = ['시퀀스번호', '현재시점_인덱스', '예측시점_인덱스', 
                   '실제_TOTALCNT', '예측_TOTALCNT', '회귀_오차',
                   '실제_구간명', '예측_구간명', '구간_정확', '신뢰도', '상태']
    
    print(results_df[display_cols].head(20).to_string(index=False))
    
    # 10. 구간별 혼동행렬 출력
    print(f"\n" + "="*50)
    print("구간별 예측 결과 (혼동행렬)")
    print("="*50)
    print("행: 실제, 열: 예측")
    print(level_distribution)
    
    # 11. 위험 사례 분석
    danger_cases = results_df[results_df['실제_TOTALCNT'] >= 1700]
    if len(danger_cases) > 0:
        print(f"\n" + "="*50)
        print(f"위험 구간(1700+) 예측 분석 ({len(danger_cases)}건)")
        print("="*50)
        danger_display_cols = ['시퀀스번호', '실제_TOTALCNT', '예측_TOTALCNT', '회귀_오차', 
                              '실제_구간명', '예측_구간명', '위험_정확', '신뢰도']
        print(danger_cases[danger_display_cols].head(10).to_string(index=False))
    
    # 12. 이상신호 분석
    anomaly_cases = results_df[results_df['실제_이상신호'] == 1]
    if len(anomaly_cases) > 0:
        print(f"\n" + "="*50)
        print(f"이상신호(1651-1682) 감지 분석 ({len(anomaly_cases)}건)")
        print("="*50)
        anomaly_display_cols = ['시퀀스번호', '실제_TOTALCNT', '예측_TOTALCNT', 
                               '실제_이상신호', '예측_이상신호', '신뢰도']
        print(anomaly_cases[anomaly_display_cols].head(10).to_string(index=False))
    
    return results_df, summary_data

# 13. 메인 실행 함수
def main():
    """메인 실행"""
    
    print("\n사용법:")
    print("1. 평가할 CSV 파일 경로 입력")
    print("2. 결과는 prediction_results 폴더에 저장됨")
    print("3. 화면에 상위 결과들이 표시됨")
    print("\n" + "-"*80)
    
    # CSV 파일 경로 입력
    csv_path = input("\n평가할 CSV 파일 경로를 입력하세요: ").strip()
    
    # 파일 존재 확인
    if not os.path.exists(csv_path):
        print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
        return
    
    # 모델 디렉토리 확인
    if not os.path.exists(model_dir):
        print(f"❌ 모델 디렉토리를 찾을 수 없습니다: {model_dir}")
        print("먼저 모델을 학습시켜 주세요.")
        return
    
    # 평가 실행
    try:
        results_df, summary_data = evaluate_and_predict_csv(csv_path)
        
        if results_df is not None:
            print("\n" + "="*80)
            print("평가 완료!")
            print("="*80)
            print(f"결과 파일 위치: {output_dir}/")
            print("- prediction_details_[timestamp].csv: 상세 예측 결과")
            print("- prediction_summary_[timestamp].csv: 성능 요약")
            print("\n주요 성능 지표:")
            print(f"- 회귀 MAE: {summary_data['회귀_MAE']:.2f}")
            print(f"- 분류 정확도: {summary_data['분류_정확도']:.3f}")
            print(f"- 위험 감지 F1: {summary_data['위험감지_F1']:.3f}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

# 14. 배치 평가 함수 (여러 파일 동시 평가)
def batch_evaluate(csv_files):
    """여러 CSV 파일을 한번에 평가"""
    
    all_results = []
    
    for csv_file in csv_files:
        print(f"\n{'='*80}")
        print(f"평가 중: {csv_file}")
        print('='*80)
        
        try:
            results_df, summary_data = evaluate_and_predict_csv(csv_file)
            if results_df is not None:
                summary_data['파일명'] = csv_file
                all_results.append(summary_data)
        except Exception as e:
            print(f"❌ 오류: {str(e)}")
            continue
    
    if all_results:
        # 전체 결과 요약
        batch_summary_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_file = os.path.join(output_dir, f'batch_evaluation_{timestamp}.csv')
        batch_summary_df.to_csv(batch_file, index=False, encoding='utf-8-sig')
        
        print(f"\n{'='*80}")
        print("배치 평가 완료")
        print('='*80)
        print(f"배치 결과: {batch_file}")
        print("\n파일별 성능 요약:")
        display_cols = ['파일명', '회귀_MAE', '분류_정확도', '위험감지_F1']
        print(batch_summary_df[display_cols].to_string(index=False))
    
    return all_results

# 실행
if __name__ == "__main__":
    main()
"""
🎯 모델별/시퀀스별 추천 및 권장사항 분석 시스템 (수정판)
====================================================
KeyError 수정 및 안정성 향상
각 조건에 따른 최적 모델 추천
시간대별/시퀀스별 권장 설정 제시
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')

def full_analysis_with_recommendations():
    """전체 분석 + 모델 추천 + 권장사항 (수정판)"""
    
    print("=" * 80)
    print("🎯 모델별/시퀀스별 추천 분석 시스템 (수정판)")
    print("=" * 80)
    
    # 1. 데이터 로드
    csv_files = ['gs.csv', 'gs.CSV', './gs.csv', './gs.CSV']
    df = None
    
    for file_path in csv_files:
        if os.path.exists(file_path):
            print(f"📁 데이터 로드: {file_path}")
            df = pd.read_csv(file_path)
            break
    
    if df is None:
        print("❌ gs.csv 파일을 찾을 수 없습니다!")
        return
    
    # 데이터 전처리
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')
    df = df.sort_values('CURRTIME').reset_index(drop=True)
    df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
    print(f"📊 분석 데이터: {len(df):,}행")
    
    # 2. 분석 설정
    sequence_lengths = [10, 20, 30, 50, 80, 100, 120, 150, 180, 200, 240, 280, 300]
    models = {
        'LSTM': {'최적시퀀스': 100, '특화': '장기패턴', '가중치': 0.25},
        'GRU': {'최적시퀀스': 60, '특화': '단기변화', '가중치': 0.20},
        'CNN_LSTM': {'최적시퀀스': 80, '특화': '복합패턴', '가중치': 0.25},
        'SpikeDetector': {'최적시퀀스': 30, '특화': '급변감지', '가중치': 0.15},
        'ExtremeNet': {'최적시퀀스': 200, '특화': '극단값예측', '가중치': 0.15}
    }
    
    # 3. 결과 저장용
    all_results = []
    
    # 헤더 추가
    all_results.append({
        'type': '=== 🎯 모델별/시퀀스별 추천 분석 결과 ===',
        'seq_len': 'analysis_range: 10~300min',
        'hour': 'analysis_range: 0~23h',
        'model': '5_models_analysis',
        'rating': '★★★(best) ~ ★(good)',
        'recommendation': 'optimal_model_per_condition',
        'total_seq': f'{len(df):,}_rows_full_analysis',
        'high_ratio': 'TOTALCNT >= 1651',
        'performance': '0~100_points',
        'notes': 'detailed_recommendations'
    })
    
    all_results.append({})  # 구분선
    
    # 4. 시퀀스별 모델 분석
    for seq_idx, seq_len in enumerate(sequence_lengths):
        print(f"\n[{seq_idx+1}/{len(sequence_lengths)}] 🔍 시퀀스 {seq_len}분 분석 중...")
        
        # 가능한 시퀀스 수
        max_sequences = len(df) - seq_len - 10
        if max_sequences <= 0:
            continue
        
        print(f"  📈 전체 시퀀스: {max_sequences:,}개")
        
        # 섹션 헤더
        all_results.append({
            'type': f'🔍_sequence_{seq_len}min_analysis',
            'seq_len': seq_len,
            'hour': 'all',
            'model': 'basic_analysis',
            'rating': '',
            'recommendation': f'{max_sequences:,}_sequences_full_analysis',
            'total_seq': max_sequences,
            'high_ratio': '',
            'performance': '',
            'notes': f'prediction_target: {seq_len}min + 10min'
        })
        
        # 전체 시퀀스 분석
        try:
            sequence_stats = analyze_all_sequences_safe(df, seq_len, max_sequences)
        except Exception as e:
            print(f"  ❌ 시퀀스 분석 오류: {e}")
            continue
        
        # 5. 모델별 추천 분석
        all_results.append({})  # 구분선
        all_results.append({
            'type': f'🤖_model_analysis_seq_{seq_len}min',
            'seq_len': '',
            'hour': '',
            'model': '',
            'rating': '',
            'recommendation': '',
            'total_seq': '',
            'high_ratio': '',
            'performance': '',
            'notes': ''
        })
        
        model_recommendations = []
        
        for model_name, model_info in models.items():
            try:
                model_analysis = analyze_model_for_sequence_safe(model_name, model_info, sequence_stats, seq_len)
                
                # 추천 등급 결정
                performance = model_analysis.get('performance_score', 70)
                if performance >= 90:
                    recommendation = '★★★_best_recommend'
                elif performance >= 80:
                    recommendation = '★★_strong_recommend'
                elif performance >= 70:
                    recommendation = '★_recommend'
                else:
                    recommendation = '△_normal'
                
                model_recommendations.append({
                    'model': model_name,
                    'performance': performance,
                    'recommendation': recommendation,
                    'analysis': model_analysis
                })
                
                all_results.append({
                    'type': 'model_analysis',
                    'seq_len': seq_len,
                    'hour': 'all',
                    'model': f"{model_name}_{model_info['특화']}",
                    'rating': recommendation,
                    'recommendation': model_analysis.get('advice', 'no_advice'),
                    'total_seq': sequence_stats.get('total_sequences', 0),
                    'high_ratio': f"{sequence_stats.get('high_ratio', 0):.1f}%",
                    'performance': f"{performance:.1f}",
                    'notes': model_analysis.get('special_notes', 'normal')
                })
                
            except Exception as e:
                print(f"  ⚠️ 모델 {model_name} 분석 오류: {e}")
                continue
        
        # 6. 시간대별 분석
        all_results.append({})  # 구분선
        all_results.append({
            'type': f'⏰_hourly_analysis_seq_{seq_len}min',
            'seq_len': '',
            'hour': '',
            'model': '',
            'rating': '',
            'recommendation': '',
            'total_seq': '',
            'high_ratio': '',
            'performance': '',
            'notes': ''
        })
        
        try:
            hourly_analysis = analyze_hourly_recommendations_safe(df, seq_len, max_sequences, model_recommendations)
            
            # 시간대별 결과 추가
            for hour_data in hourly_analysis:
                if hour_data.get('total_sequences', 0) > 0:
                    all_results.append(hour_data)
                    
        except Exception as e:
            print(f"  ⚠️ 시간대별 분석 오류: {e}")
        
        # 7. 시퀀스별 최종 추천
        if model_recommendations:
            best_models = sorted(model_recommendations, key=lambda x: x['performance'], reverse=True)[:2]
            
            all_results.append({})  # 구분선
            all_results.append({
                'type': f'🏆_final_recommendation_seq_{seq_len}min',
                'seq_len': seq_len,
                'hour': 'all',
                'model': f"1st_{best_models[0]['model']}_2nd_{best_models[1]['model'] if len(best_models) > 1 else 'N/A'}",
                'rating': f"{best_models[0]['recommendation']}_{best_models[1]['recommendation'] if len(best_models) > 1 else 'N/A'}",
                'recommendation': get_final_recommendation_safe(seq_len, best_models, sequence_stats),
                'total_seq': sequence_stats.get('total_sequences', 0),
                'high_ratio': f"{sequence_stats.get('high_ratio', 0):.1f}%",
                'performance': f"{best_models[0]['performance']:.1f}",
                'notes': get_special_notes_safe(seq_len, sequence_stats)
            })
        
        all_results.append({})  # 시퀀스 구분선
    
    # 8. 전체 추천 요약
    all_results.append({
        'type': '=== 🌟 overall_final_recommendations ===',
        'seq_len': '',
        'hour': '',
        'model': '',
        'rating': '',
        'recommendation': '',
        'total_seq': '',
        'high_ratio': '',
        'performance': '',
        'notes': ''
    })
    
    try:
        overall_recommendations = generate_overall_recommendations_safe(all_results)
        all_results.extend(overall_recommendations)
    except Exception as e:
        print(f"⚠️ 전체 추천 생성 오류: {e}")
    
    # 9. 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"model_sequence_recommendations_{timestamp}.csv"
    
    try:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 분석 완료!")
        print(f"📁 결과 파일: {output_file}")
        print(f"📊 총 결과: {len(results_df):,}행")
        print(f"🎯 {len(sequence_lengths)}개 시퀀스 × 5개 모델 × 24시간 분석 완료")
        
    except Exception as e:
        print(f"❌ 파일 저장 오류: {e}")

def analyze_all_sequences_safe(df, seq_len, max_sequences):
    """전체 시퀀스 기본 분석 (안전 버전)"""
    
    try:
        high_count = 0
        max_values = []
        volatility_scores = []
        trends = []
        
        # 샘플링 (너무 많으면)
        if max_sequences > 10000:
            step = max_sequences // 5000
            indices = range(seq_len, max_sequences + seq_len, step)
            print(f"  📊 샘플링 분석: {len(indices):,}개 ({step}개마다)")
        else:
            indices = range(seq_len, max_sequences + seq_len)
            print(f"  📊 전체 분석: {len(indices):,}개")
        
        for i in indices:
            try:
                seq_data = df.iloc[i-seq_len:i]['TOTALCNT'].values
                seq_max = np.max(seq_data)
                seq_std = np.std(seq_data)
                
                max_values.append(seq_max)
                volatility_scores.append(seq_std)
                
                if seq_max >= 1651:
                    high_count += 1
                
                # 간단한 추세 분석
                if len(seq_data) > 1:
                    if seq_data[-1] > seq_data[0]:
                        trends.append('증가')
                    elif seq_data[-1] < seq_data[0]:
                        trends.append('감소')
                    else:
                        trends.append('안정')
                else:
                    trends.append('안정')
                    
            except Exception as e:
                print(f"    ⚠️ 인덱스 {i} 처리 오류: {e}")
                continue
        
        total_analyzed = len(max_values)
        if total_analyzed == 0:
            return {
                'total_sequences': 0,
                'high_sequences': 0,
                'high_ratio': 0,
                'avg_max': 0,
                'avg_volatility': 0,
                'increase_trends': 0,
                'decrease_trends': 0,
                'stable_trends': 0
            }
        
        return {
            'total_sequences': total_analyzed,
            'high_sequences': high_count,
            'high_ratio': high_count / total_analyzed * 100,
            'avg_max': np.mean(max_values),
            'avg_volatility': np.mean(volatility_scores),
            'increase_trends': trends.count('증가'),
            'decrease_trends': trends.count('감소'),
            'stable_trends': trends.count('안정')
        }
        
    except Exception as e:
        print(f"  ❌ 시퀀스 분석 전체 오류: {e}")
        return {
            'total_sequences': 0,
            'high_sequences': 0,
            'high_ratio': 0,
            'avg_max': 0,
            'avg_volatility': 0,
            'increase_trends': 0,
            'decrease_trends': 0,
            'stable_trends': 0
        }

def analyze_model_for_sequence_safe(model_name, model_info, sequence_stats, seq_len):
    """모델별 시퀀스 분석 (안전 버전)"""
    
    try:
        optimal_seq = model_info.get('최적시퀀스', 100)
        specialty = model_info.get('특화', '범용')
        
        # 기본 성능 계산
        base_performance = 70
        
        # 시퀀스 길이 적합도
        length_diff = abs(seq_len - optimal_seq)
        length_penalty = length_diff / optimal_seq * 30
        sequence_score = max(40, base_performance - length_penalty)
        
        # 모델별 특화 점수
        specialty_score = 0
        advice = ""
        special_notes = ""
        
        if model_name == 'ExtremeNet':
            # 고값 감지 특화
            high_ratio = sequence_stats.get('high_ratio', 0)
            specialty_score = high_ratio * 1.5
            boost_conditions = int(sequence_stats.get('high_sequences', 0) * 0.3)
            advice = f"high_value_detection_specialized_{high_ratio:.1f}%_high_sequences"
            special_notes = f"V6.7_boosting_{boost_conditions}_expected"
            
        elif model_name == 'SpikeDetector':
            # 급변 감지 특화
            avg_volatility = sequence_stats.get('avg_volatility', 0)
            specialty_score = min(30, avg_volatility / 2)
            advice = f"spike_detection_specialized_volatility_{avg_volatility:.1f}"
            special_notes = f"volatility_based_boosting_active"
            
        elif model_name == 'LSTM':
            # 장기 패턴 특화
            if seq_len >= 100:
                specialty_score = 20
            elif seq_len >= 80:
                specialty_score = 15
            else:
                specialty_score = 5
            advice = f"long_pattern_specialized_optimal_{optimal_seq}min_current_{seq_len}min"
            special_notes = f"best_performance_at_100min+"
            
        elif model_name == 'GRU':
            # 단기 변화 특화
            if 50 <= seq_len <= 80:
                specialty_score = 18
            elif 30 <= seq_len <= 100:
                specialty_score = 12
            else:
                specialty_score = 5
            advice = f"short_change_detection_optimal_{optimal_seq}min_current_{seq_len}min"
            special_notes = f"highest_efficiency_at_50-80min"
            
        elif model_name == 'CNN_LSTM':
            # 복합 패턴 특화
            if 70 <= seq_len <= 100:
                specialty_score = 17
            elif 50 <= seq_len <= 120:
                specialty_score = 12
            else:
                specialty_score = 7
            advice = f"complex_pattern_recognition_optimal_{optimal_seq}min_current_{seq_len}min"
            special_notes = f"2D_pattern_conversion_high_accuracy"
        
        # 최종 성능 점수
        final_performance = sequence_score + specialty_score
        final_performance = max(40, min(95, final_performance))
        
        return {
            'performance_score': final_performance,
            'advice': advice,
            'special_notes': special_notes,
            'sequence_fitness': sequence_score,
            'specialty_score': specialty_score
        }
        
    except Exception as e:
        print(f"    ⚠️ 모델 {model_name} 분석 오류: {e}")
        return {
            'performance_score': 70,
            'advice': 'analysis_error',
            'special_notes': str(e),
            'sequence_fitness': 70,
            'specialty_score': 0
        }

def analyze_hourly_recommendations_safe(df, seq_len, max_sequences, model_recommendations):
    """시간대별 모델 추천 분석 (안전 버전)"""
    
    hourly_results = []
    
    try:
        # 시간대별 데이터 수집
        hourly_stats = {}
        for hour in range(24):
            hourly_stats[hour] = {
                'sequence_count': 0,
                'high_count': 0,
                'max_values': [],
                'volatility_values': []
            }
        
        # 샘플링해서 시간대별 집계
        if max_sequences > 5000:
            step = max(1, max_sequences // 2000)
            indices = range(seq_len, max_sequences + seq_len, step)
        else:
            indices = range(seq_len, max_sequences + seq_len)
        
        for i in indices:
            try:
                hour = df.iloc[i]['CURRTIME'].hour
                seq_data = df.iloc[i-seq_len:i]['TOTALCNT'].values
                seq_max = np.max(seq_data)
                seq_std = np.std(seq_data)
                
                hourly_stats[hour]['sequence_count'] += 1
                hourly_stats[hour]['max_values'].append(seq_max)
                hourly_stats[hour]['volatility_values'].append(seq_std)
                
                if seq_max >= 1651:
                    hourly_stats[hour]['high_count'] += 1
                    
            except Exception as e:
                continue
        
        # 시간대별 추천 생성
        for hour in range(24):
            stats = hourly_stats[hour]
            
            if stats['sequence_count'] == 0:
                continue
            
            try:
                high_ratio = stats['high_count'] / stats['sequence_count'] * 100
                avg_max = np.mean(stats['max_values']) if stats['max_values'] else 0
                avg_volatility = np.mean(stats['volatility_values']) if stats['volatility_values'] else 0
                
                # 시간대별 최적 모델 결정
                if high_ratio > 15:  # 고위험 시간대
                    best_model = 'ExtremeNet'
                    recommendation = '★★★_high_value_detection_required'
                    advice = f"high_risk_time_ExtremeNet_SpikeDetector_combination"
                elif avg_volatility > 40:  # 고변동성 시간대
                    best_model = 'SpikeDetector'
                    recommendation = '★★_spike_detection_needed'
                    advice = f"high_volatility_SpikeDetector_main_GRU_support"
                elif 2 <= hour <= 5:  # 새벽 시간대
                    best_model = 'LSTM'
                    recommendation = '★★_long_pattern_stable'
                    advice = f"stable_time_LSTM_long_prediction_optimal"
                else:  # 일반 시간대
                    best_model = 'CNN_LSTM'
                    recommendation = '★_general_use'
                    advice = f"normal_time_CNN_LSTM_complex_pattern"
                
                special_notes = get_hourly_special_notes_safe(hour, high_ratio, avg_volatility)
                
                hourly_results.append({
                    'type': 'hourly_analysis',
                    'seq_len': seq_len,
                    'hour': f"{hour:02d}h",
                    'model': best_model,
                    'rating': recommendation,
                    'recommendation': advice,
                    'total_sequences': stats['sequence_count'],
                    'high_ratio': f"{high_ratio:.1f}%",
                    'performance': f"{avg_max:.0f}",
                    'notes': special_notes
                })
                
            except Exception as e:
                print(f"    ⚠️ {hour}시 분석 오류: {e}")
                continue
    
    except Exception as e:
        print(f"  ❌ 시간대별 분석 전체 오류: {e}")
    
    return hourly_results

def get_hourly_special_notes_safe(hour, high_ratio, avg_volatility):
    """시간대별 특이사항 생성 (안전 버전)"""
    
    try:
        notes = []
        
        if high_ratio > 20:
            notes.append("⚠️_high_risk")
        elif high_ratio < 5:
            notes.append("✅_safe")
        
        if avg_volatility > 50:
            notes.append("📈_high_volatility")
        elif avg_volatility < 20:
            notes.append("📊_stable")
        
        # 시간대별 특성
        if 0 <= hour <= 5:
            notes.append("🌙_dawn_time")
        elif 6 <= hour <= 11:
            notes.append("🌅_morning_time")
        elif 12 <= hour <= 17:
            notes.append("☀️_afternoon_time")
        else:
            notes.append("🌆_evening_time")
        
        return "_".join(notes) if notes else "normal"
        
    except Exception:
        return "analysis_error"

def get_final_recommendation_safe(seq_len, best_models, sequence_stats):
    """시퀀스별 최종 추천사항 (안전 버전)"""
    
    try:
        recommendations = []
        
        # 시퀀스 길이별 추천
        if seq_len <= 30:
            recommendations.append("short_prediction_SpikeDetector_active_use")
        elif seq_len <= 80:
            recommendations.append("medium_prediction_GRU_CNN_LSTM_combination")
        elif seq_len <= 150:
            recommendations.append("long_prediction_LSTM_main_use")
        else:
            recommendations.append("ultra_long_prediction_LSTM_ExtremeNet_combination")
        
        # 고값 비율별 추천
        high_ratio = sequence_stats.get('high_ratio', 0)
        if high_ratio > 15:
            recommendations.append("high_value_frequent_ExtremeNet_required")
        elif high_ratio > 10:
            recommendations.append("high_value_caution_ExtremeNet_support")
        
        # 변동성별 추천
        avg_volatility = sequence_stats.get('avg_volatility', 0)
        if avg_volatility > 40:
            recommendations.append("high_volatility_SpikeDetector_enhanced")
        
        return "_".join(recommendations) if recommendations else "standard_recommendation"
        
    except Exception:
        return "recommendation_generation_error"

def get_special_notes_safe(seq_len, sequence_stats):
    """특이사항 생성 (안전 버전)"""
    
    try:
        notes = []
        
        # 시퀀스 길이 평가
        if seq_len == 100:
            notes.append("🔥_currently_used_length")
        elif seq_len == 280:
            notes.append("⭐_previous_analysis_optimal_length")
        
        # 고값 비율 평가
        high_ratio = sequence_stats.get('high_ratio', 0)
        if high_ratio > 20:
            notes.append("🚨_very_high_value_ratio")
        elif high_ratio > 15:
            notes.append("⚠️_high_value_ratio")
        elif high_ratio < 5:
            notes.append("✅_safe_value_ratio")
        
        return "_".join(notes) if notes else "standard"
        
    except Exception:
        return "notes_generation_error"

def generate_overall_recommendations_safe(all_results):
    """전체 추천 요약 생성 (안전 버전)"""
    
    summary_results = []
    
    try:
        # 최고 성능 시퀀스 찾기
        model_results = [r for r in all_results if r.get('type') == 'model_analysis']
        
        if model_results:
            # 성능 점수 기준 상위 추천
            best_performances = []
            for result in model_results:
                try:
                    if result.get('performance'):
                        score = float(str(result['performance']).replace('점', ''))
                        best_performances.append((score, result))
                except:
                    continue
            
            best_performances.sort(reverse=True)
            
            summary_results.append({
                'type': '🥇_best_performance_combinations',
                'seq_len': '',
                'hour': '',
                'model': '',
                'rating': '',
                'recommendation': '',
                'total_seq': '',
                'high_ratio': '',
                'performance': '',
                'notes': ''
            })
            
            # 상위 5개 추천
            for i, (score, result) in enumerate(best_performances[:5]):
                rank_emoji = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][i]
                summary_results.append({
                    'type': f'{rank_emoji}_recommendation',
                    'seq_len': result.get('seq_len', ''),
                    'hour': result.get('hour', ''),
                    'model': result.get('model', ''),
                    'rating': result.get('rating', ''),
                    'recommendation': f"{score:.1f}points_{result.get('recommendation', '')}",
                    'total_seq': result.get('total_seq', ''),
                    'high_ratio': result.get('high_ratio', ''),
                    'performance': f"{score:.1f}",
                    'notes': result.get('notes', '')
                })
        
        # 종합 권장사항
        summary_results.append({})
        summary_results.append({
            'type': '📋_comprehensive_recommendations',
            'seq_len': 'situation_based_optimal_selection',
            'hour': 'hourly_model_switching',
            'model': 'ensemble_combination_recommended',
            'rating': '★★★',
            'recommendation': 'ExtremeNet_for_high_value_detection_LSTM_for_stability',
            'total_seq': 'full_data_based',
            'high_ratio': 'target_15%_or_higher',
            'performance': 'target_90_points_or_higher',
            'notes': 'real_time_monitoring_for_model_switching'
        })
        
    except Exception as e:
        print(f"⚠️ 전체 추천 생성 오류: {e}")
        summary_results.append({
            'type': 'summary_generation_error',
            'seq_len': '',
            'hour': '',
            'model': '',
            'rating': '',
            'recommendation': str(e),
            'total_seq': '',
            'high_ratio': '',
            'performance': '',
            'notes': ''
        })
    
    return summary_results

if __name__ == "__main__":
    print("🎯 모델별/시퀀스별 추천 분석을 시작합니다...")
    print("⏰ 전체 데이터 분석으로 시간이 소요될 수 있습니다.")
    
    start_analysis = input("분석을 시작하시겠습니까? (y/N): ")
    if start_analysis.lower() == 'y':
        full_analysis_with_recommendations()
    else:
        print("❌ 분석을 취소했습니다.")
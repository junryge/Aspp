def predict_sequence(self, sequence):
    """단일 시퀀스 예측 (30분 버전) - 점프 감지 강화"""
    past_values = sequence['past_30min_values']
    physics = sequence['physics_features']
    
    # 30분 데이터 분석
    recent_5 = past_values[-5:]
    recent_10 = past_values[-10:]
    first_10 = past_values[:10]
    middle_10 = past_values[10:20]
    last_10 = past_values[-10:]
    
    # 핵심 지표 계산
    long_trend = np.mean(last_10) - np.mean(first_10)
    mid_trend = np.mean(last_10) - np.mean(middle_10)
    acceleration = mid_trend - (np.mean(middle_10) - np.mean(first_10))
    
    max_val = max(past_values)
    min_val = min(past_values)
    mean_recent = np.mean(recent_5)
    mean_last10 = np.mean(last_10)
    
    # 변동성 지표
    std_first = np.std(first_10) if np.std(first_10) > 0 else 1
    std_last = np.std(last_10)
    volatility_change = std_last / std_first
    
    # 상승 카운트
    rising_count = sum(1 for i in range(1, len(past_values)) if past_values[i] > past_values[i-1])
    count_300plus = sum(1 for v in past_values if v >= 300)
    
    # ========================================
    # 1단계: 270-280 점프 위험 구간 특별 처리 (최우선)
    # ========================================
    if 270 <= max_val <= 280:
        # 여러 신호 체크
        jump_signals = 0
        
        # 신호 1: 가속도
        if acceleration > 2:
            jump_signals += 2
        elif acceleration > 1:
            jump_signals += 1
            
        # 신호 2: 상승 추세
        if long_trend > 5:
            jump_signals += 2
        elif long_trend > 2:
            jump_signals += 1
            
        # 신호 3: 변동성 증가
        if volatility_change > 1.5:
            jump_signals += 1
            
        # 신호 4: 연속 상승
        if rising_count > 20:  # 30개 중 20개 이상 상승
            jump_signals += 2
        elif rising_count > 15:
            jump_signals += 1
            
        # 신호 5: 최근 급등
        if mean_recent > 275:
            jump_signals += 3  # 강한 신호
        elif mean_recent > 272:
            jump_signals += 2
            
        # 신호 6: BRIDGE_TIME (물리 특징)
        if physics[3] > 4.5:  # BRIDGE_TIME
            jump_signals += 2
        elif physics[3] > 4:
            jump_signals += 1
            
        # 점프 예측 결정
        if jump_signals >= 4:  # 충분한 신호
            selected_model = "Model2"
            # 점프 강도에 따른 예측
            if jump_signals >= 7:
                predicted = mean_recent * 1.15  # 강한 점프
            elif jump_signals >= 5:
                predicted = mean_recent * 1.12  # 중간 점프
            else:
                predicted = mean_recent * 1.10  # 약한 점프
                
            # 최소 300 보장
            if predicted < 300 and mean_recent > 273:
                predicted = 302  # 점프 최소값
                
            return predicted, selected_model
    
    # ========================================
    # 2단계: 260-270 사전 경고 구간
    # ========================================
    if 260 <= max_val <= 270:
        if acceleration > 3 or (long_trend > 8 and volatility_change > 2):
            selected_model = "Model2"
            predicted = mean_recent * 1.08
            return predicted, selected_model
    
    # ========================================
    # 3단계: 이미 극단값 구간 (300+)
    # ========================================
    if min(recent_5) > 310:
        selected_model = "Model2"
        # 극단값 유지/상승
        if acceleration > 0:
            predicted = mean_recent * 1.04
        else:
            predicted = mean_recent * 1.02
        return predicted, selected_model
    
    if mean_recent > 295:
        selected_model = "Model2"
        predicted = mean_recent * 1.03
        return predicted, selected_model
    
    # ========================================
    # 4단계: 확실한 안정 구간
    # ========================================
    if max_val < 250 and std_last < 20:
        selected_model = "Model1"
        predicted = mean_recent * 0.98
        return predicted, selected_model
    
    # ========================================
    # 5단계: 경계 구간 (250-270) - 세밀한 판단
    # ========================================
    if 250 <= max_val <= 270:
        # 위험 점수 계산
        risk_score = 0
        
        # 추세 점수
        if long_trend > 15:
            risk_score += 30
        elif long_trend > 8:
            risk_score += 20
        elif long_trend > 3:
            risk_score += 10
        elif long_trend < -10:
            risk_score -= 10
            
        # 가속도 점수
        if acceleration > 5:
            risk_score += 25
        elif acceleration > 2:
            risk_score += 15
        elif acceleration < -3:
            risk_score -= 10
            
        # 변동성 점수
        if volatility_change > 2.5:
            risk_score += 20
        elif volatility_change > 1.5:
            risk_score += 10
            
        # 현재 레벨 점수
        if mean_recent > 265:
            risk_score += 25
        elif mean_recent > 260:
            risk_score += 15
        elif mean_recent < 240:
            risk_score -= 15
            
        # 물리 특징 점수
        if physics[3] > 5:  # BRIDGE_TIME
            risk_score += 20
        elif physics[3] > 4:
            risk_score += 10
            
        if physics[5] > 0.5:  # consecutive_300_prob
            risk_score += 15
        elif physics[5] > 0.3:
            risk_score += 8
            
        # 모델 선택
        if risk_score >= 50:
            selected_model = "Model2"
            if risk_score >= 70:
                predicted = mean_recent * 1.08
            else:
                predicted = mean_recent * 1.05
        else:
            selected_model = "Model1"
            if risk_score <= 10:
                predicted = mean_recent * 0.97
            else:
                predicted = mean_recent * 0.99
                
        # 경계 구간 특별 조정
        if 265 <= mean_recent <= 275 and acceleration > 1:
            predicted *= 1.03  # 추가 보정
            
        return predicted, selected_model
    
    # ========================================
    # 6단계: 나머지 케이스 (기본 로직)
    # ========================================
    if count_300plus > 8:  # 30개 중 8개로 낮춤 (더 민감)
        selected_model = "Model2"
        predicted = mean_recent * 1.04
    else:
        selected_model = "Model1"
        predicted = mean_recent * 0.98
    
    # ========================================
    # 7단계: 최종 보정 및 제한
    # ========================================
    
    # Model2 과대예측 방지
    if selected_model == "Model2":
        # 하락 추세에서 과대예측 방지
        if long_trend < -30 and predicted > 300:
            predicted = mean_recent * 1.01  # 매우 보수적
            
        # 최대값 대비 과도한 예측 방지
        if predicted > max_val * 1.4:
            predicted = max_val * 1.25
            
        # 비현실적 예측 방지
        if predicted > 400 and max_val < 350:
            predicted = max_val * 1.15
    
    # Model1 과소예측 방지
    if selected_model == "Model1":
        # 상승 추세에서 과소예측 방지
        if long_trend > 20 and predicted < mean_recent * 0.95:
            predicted = mean_recent * 0.98
    
    # 최종 범위 제한
    predicted = max(50, min(450, predicted))  # 50-450 범위
    
    return predicted, selected_model
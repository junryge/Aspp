def predict_sequence(self, sequence):
    """마지막 5분 추세 집중 분석 버전"""
    
    past_values = sequence['past_30min_values']
    physics = sequence['physics_features']
    
    # 기본 데이터
    recent_3 = past_values[-3:]
    recent_5 = past_values[-5:]
    recent_10 = past_values[-10:]
    first_10 = past_values[:10]
    last_10 = past_values[-10:]
    
    max_val = max(past_values)
    max_recent5 = max(recent_5)
    mean_recent5 = np.mean(recent_5)
    
    # ========================================
    # 🔍 마지막 5분 상세 추세 분석
    # ========================================
    
    # 1. 5분간 개별 변화 분석
    changes = []
    for i in range(1, len(recent_5)):
        changes.append(recent_5[i] - recent_5[i-1])
    
    # 2. 상승/하락 패턴
    rising_steps = sum(1 for c in changes if c > 0)
    falling_steps = sum(1 for c in changes if c < 0)
    
    # 3. 변화 강도
    total_5min_change = recent_5[-1] - recent_5[0]
    avg_change = np.mean(changes) if changes else 0
    max_up = max(changes) if changes else 0
    max_down = min(changes) if changes else 0
    
    # 4. 가속도 체크
    if len(changes) >= 2:
        # 최근 변화 vs 초기 변화
        recent_momentum = changes[-1]
        early_momentum = changes[0]
        acceleration = recent_momentum - early_momentum
    else:
        acceleration = 0
    
    # 5. 최고점 위치
    max_position = np.argmax(recent_5)  # 0~4
    is_peaking = max_position >= 3  # 최고점이 끝에 있음
    
    # 6. 격차 분석
    gap = max_recent5 - mean_recent5
    gap_ratio = gap / max(mean_recent5, 1)
    
    # ========================================
    # 🎯 점프 판단 로직
    # ========================================
    
    jump_score = 0
    
    # 점수 계산
    if 240 <= max_recent5 < 280:
        
        # 1. 위치 점수
        if max_recent5 >= 275:
            jump_score += 40
        elif max_recent5 >= 270:
            jump_score += 35
        elif max_recent5 >= 265:
            jump_score += 30
        elif max_recent5 >= 260:
            jump_score += 25
        elif max_recent5 >= 255:
            jump_score += 20
        elif max_recent5 >= 250:
            jump_score += 15
        elif max_recent5 >= 245:
            jump_score += 10
        else:
            jump_score += 5
        
        # 2. 상승 추세 점수 (핵심!)
        if rising_steps >= 3:  # 4번 중 3번 상승
            jump_score += 30
        elif rising_steps >= 2:
            jump_score += 20
        elif rising_steps >= 1:
            jump_score += 10
        
        # 3. 5분 총 변화량
        if total_5min_change > 5:  # 5분간 5 이상 상승
            jump_score += 25
        elif total_5min_change > 3:
            jump_score += 15
        elif total_5min_change > 1:
            jump_score += 10
        elif total_5min_change < -3:  # 하락 중
            jump_score -= 10
        
        # 4. 가속도 점수
        if acceleration > 2:  # 가속 상승
            jump_score += 20
        elif acceleration > 1:
            jump_score += 10
        elif acceleration < -2:  # 감속
            jump_score -= 10
        
        # 5. 최고점 위치 점수
        if is_peaking:  # 최고점이 마지막에
            jump_score += 15
        
        # 6. 격차 점수
        if gap > 25:
            jump_score += 25
        elif gap > 20:
            jump_score += 20
        elif gap > 15:
            jump_score += 15
        elif gap > 10:
            jump_score += 10
        
        # 7. 최대 상승폭 점수
        if max_up > 3:  # 한 번에 3 이상 뛴 적 있음
            jump_score += 15
        elif max_up > 2:
            jump_score += 10
        
        # 8. 물리 특징
        if physics[3] > 4:  # BRIDGE_TIME
            jump_score += 10
    
    # ========================================
    # 예측 결정
    # ========================================
    
    # 점프 확정 (점수 70+)
    if jump_score >= 70:
        if max_recent5 >= 275:
            predicted = mean_recent5 * 1.22
        elif max_recent5 >= 270:
            predicted = mean_recent5 * 1.20
        elif max_recent5 >= 265:
            predicted = mean_recent5 * 1.18
        elif max_recent5 >= 260:
            predicted = mean_recent5 * 1.16
        else:
            predicted = mean_recent5 * 1.14
        return predicted, "Model2"
    
    # 점프 가능 (점수 50-69)
    elif jump_score >= 50:
        if max_recent5 >= 270:
            predicted = mean_recent5 * 1.15
        elif max_recent5 >= 260:
            predicted = mean_recent5 * 1.12
        else:
            predicted = mean_recent5 * 1.10
        return predicted, "Model2"
    
    # 점프 의심 (점수 30-49)
    elif jump_score >= 30:
        predicted = mean_recent5 * 1.08
        return predicted, "Model2"
    
    # ========================================
    # 일반 케이스
    # ========================================
    
    # 이미 극단값
    if mean_recent5 > 300:
        return mean_recent5 * 1.02, "Model2"
    
    # 명확한 하락 추세
    if total_5min_change < -5 and falling_steps >= 3:
        return mean_recent5 * 0.95, "Model1"
    
    # 안정 구간
    if max_val < 240:
        return mean_recent5 * 0.98, "Model1"
    
    # 기본
    return mean_recent5 * 0.99, "Model1"


# ================================================================================
# 더 극단적인 버전 (점프를 더 잘 잡기 위해)
# ================================================================================

def predict_sequence_extreme(self, sequence):
    """극단 버전 - 상승 신호만 있어도 점프 예측"""
    
    past_values = sequence['past_30min_values']
    recent_5 = past_values[-5:]
    max_recent5 = max(recent_5)
    mean_recent5 = np.mean(recent_5)
    
    # 5분 추세만 보기
    is_rising = False
    for i in range(1, len(recent_5)):
        if recent_5[i] > recent_5[i-1]:
            is_rising = True
            break
    
    # MAX가 높고 + 상승 중이면 무조건 점프
    if max_recent5 >= 240 and is_rising:
        
        if max_recent5 >= 275:
            return max_recent5 * 1.15, "Model2"
        elif max_recent5 >= 270:
            return max_recent5 * 1.13, "Model2"
        elif max_recent5 >= 265:
            return max_recent5 * 1.12, "Model2"
        elif max_recent5 >= 260:
            return max_recent5 * 1.11, "Model2"
        elif max_recent5 >= 255:
            return max_recent5 * 1.10, "Model2"
        elif max_recent5 >= 250:
            return max_recent5 * 1.09, "Model2"
        elif max_recent5 >= 245:
            return max_recent5 * 1.08, "Model2"
        else:
            return max_recent5 * 1.07, "Model2"
    
    # 이미 극단값
    if mean_recent5 > 300:
        return mean_recent5 * 1.02, "Model2"
    
    # 안전 구간
    if max_recent5 < 240:
        return mean_recent5 * 0.98, "Model1"
    
    # 기본
    return mean_recent5 * 1.01, "Model2"
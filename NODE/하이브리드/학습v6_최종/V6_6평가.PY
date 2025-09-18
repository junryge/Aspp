python"""
V8.0 시계열 트렌드 기반 극단값 예측
핵심: 100개 시퀀스의 패턴을 분석하여 극단값 예측 전략 선택
"""

class TrendBasedExtremePredictor:
    def __init__(self):
        self.seq_len = 100
        self.extreme_threshold = 1650
        
    def analyze_sequence_trend(self, sequence_data):
        """100개 시퀀스의 트렌드 분석"""
        # 최대값 확인
        max_value = np.max(sequence_data)
        
        # 트렌드 계산 (선형 회귀)
        x = np.arange(len(sequence_data))
        z = np.polyfit(x, sequence_data, 1)
        slope = z[0]  # 기울기
        
        # 최근 20개 vs 처음 20개 비교
        recent_mean = np.mean(sequence_data[-20:])
        early_mean = np.mean(sequence_data[:20])
        trend_ratio = recent_mean / early_mean
        
        # 변동성 계산
        volatility = np.std(sequence_data[-30:])
        
        return {
            'max_value': max_value,
            'slope': slope,
            'trend_ratio': trend_ratio,
            'volatility': volatility,
            'recent_mean': recent_mean
        }
    
    def select_prediction_strategy(self, trend_info, m14b, m14a):
        """트렌드에 따른 예측 전략 선택"""
        
        max_val = trend_info['max_value']
        slope = trend_info['slope']
        trend_ratio = trend_info['trend_ratio']
        
        # 케이스 1: 극단값 상승 패턴 (최대값 1650+ & 상승 트렌드)
        if max_val >= 1650 and slope > 0 and trend_ratio > 1.05:
            return 'EXTREME_UP'
            
        # 케이스 2: 극단값 유지 패턴 (최대값 1650+ & 안정)
        elif max_val >= 1650 and abs(slope) < 1 and 0.95 <= trend_ratio <= 1.05:
            return 'EXTREME_STABLE'
            
        # 케이스 3: 극단값 하락 패턴 (최대값 1650+ & 하락 트렌드)
        elif max_val >= 1650 and slope < -1 and trend_ratio < 0.95:
            return 'EXTREME_DOWN'
            
        # 케이스 4: 급상승 패턴 (아직 극단값 아니지만 빠르게 상승)
        elif slope > 2 and trend_ratio > 1.1:
            return 'RAPID_UP'
            
        # 케이스 5: 일반 패턴
        else:
            return 'NORMAL'
    
    def predict_extreme(self, sequence_data, m14b, m14a, base_predictions):
        """트렌드 기반 극단값 예측"""
        
        # 1. 시퀀스 트렌드 분석
        trend_info = self.analyze_sequence_trend(sequence_data)
        strategy = self.select_prediction_strategy(trend_info, m14b, m14a)
        
        # 2. 전략별 예측
        if strategy == 'EXTREME_UP':
            # 극단 상승: 매우 공격적 예측
            if m14b > 500:
                pred = max(1950, trend_info['recent_mean'] * 1.3)
            elif m14b > 450:
                pred = max(1850, trend_info['recent_mean'] * 1.25)
            else:
                pred = max(1750, trend_info['recent_mean'] * 1.2)
                
        elif strategy == 'EXTREME_STABLE':
            # 극단 유지: 현재 수준 유지 + 알파
            pred = max(1700, trend_info['recent_mean'] * 1.05)
            if m14b > 450:
                pred = max(pred, 1750)
                
        elif strategy == 'EXTREME_DOWN':
            # 극단 하락: 보수적이지만 여전히 높게
            pred = max(1650, trend_info['recent_mean'] * 0.98)
            # M14B가 여전히 높으면 하한선 보장
            if m14b > 450:
                pred = max(pred, 1700)
                
        elif strategy == 'RAPID_UP':
            # 급상승: 선제적 극단값 예측
            pred = max(1700, trend_info['recent_mean'] * 1.15)
            if slope > 3:  # 매우 가파른 상승
                pred = pred * 1.1
                
        else:  # NORMAL
            # 일반: 기존 앙상블 사용
            pred = np.mean(base_predictions)
        
        # 3. 황금 패턴 보정
        if m14b > 300 and m14a < 80:
            pred = pred * 1.15
        
        # 4. 최종 클리핑
        pred = np.clip(pred, 1000, 2100)
        
        return pred, strategy

    def evaluate_with_trend(self, test_df):
        """트렌드 기반 평가"""
        
        predictions = []
        strategies = []
        
        for i in range(100, len(test_df)):
            # 100개 시퀀스 추출
            sequence = test_df['TOTALCNT'].iloc[i-100:i].values
            m14b = test_df['M14AM14B'].iloc[i]
            m14a = test_df['M14AM10A'].iloc[i]
            
            # 기존 모델 예측값들 (있다면)
            base_preds = []
            for col in ['ExtremeNet_예측', 'PatchTST_예측', 'StableLSTM_예측']:
                if col in test_df.columns:
                    base_preds.append(test_df[col].iloc[i])
            
            if len(base_preds) == 0:
                base_preds = [test_df['TOTALCNT'].iloc[i] * 1.05]  # 기본값
            
            # 트렌드 기반 예측
            pred, strategy = self.predict_extreme(sequence, m14b, m14a, base_preds)
            predictions.append(pred)
            strategies.append(strategy)
        
        # 결과 저장
        result_df = test_df.iloc[100:].copy()
        result_df['트렌드예측'] = predictions
        result_df['예측전략'] = strategies
        
        # 성능 평가
        high_mask = result_df['TOTALCNT'] >= 1700
        if high_mask.any():
            high_df = result_df[high_mask]
            hit_count = (high_df['트렌드예측'] >= 1700).sum()
            print(f"\n🎯 트렌드 기반 1700+ 적중률: {hit_count}/{len(high_df)} "
                  f"({hit_count/len(high_df)*100:.1f}%)")
            
            # 전략별 분석
            for strategy in ['EXTREME_UP', 'EXTREME_STABLE', 'EXTREME_DOWN', 'RAPID_UP']:
                strategy_mask = high_df['예측전략'] == strategy
                if strategy_mask.any():
                    strategy_df = high_df[strategy_mask]
                    strategy_hit = (strategy_df['트렌드예측'] >= 1700).sum()
                    print(f"  {strategy}: {strategy_hit}/{len(strategy_df)} "
                          f"({strategy_hit/len(strategy_df)*100:.1f}%)")
        
        return result_df

# 메인 실행 부분에 추가
def enhanced_extreme_prediction(test_file):
    """강화된 극단값 예측"""
    
    # 데이터 로드
    df = pd.read_csv(test_file)
    
    # 트렌드 기반 예측기 생성
    trend_predictor = TrendBasedExtremePredictor()
    
    # 평가 실행
    result = trend_predictor.evaluate_with_trend(df)
    
    # 저장
    result.to_csv('v8_trend_extreme_predictions.csv', index=False)
    
    print("\n✅ 핵심 개선:")
    print("  1. 100개 시퀀스 최대값 1650+ → 극단 전략")
    print("  2. 상승 트렌드 → 공격적 예측 (1950+)")
    print("  3. 안정 트렌드 → 현재 수준 유지")
    print("  4. 하락 트렌드 → 보수적 but 하한선")
    print("  5. 급상승 패턴 → 선제적 극단값")
    
    return result
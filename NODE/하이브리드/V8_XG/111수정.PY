actual_value = df['TOTALCNT'].iloc[i]
        
        # Feature 생성
        features = create_single_prediction_features(seq_m14b, seq_m10a, seq_m16, seq_totalcnt)
        X_pred = pd.DataFrame([features])
        
        # 예측
        prediction = model.predict(X_pred)[0]
        
        # 예측 보정 (1650+ 과소 예측 방지)
        current_totalcnt = seq_totalcnt[-1]
        recent_5_avg = np.mean(seq_totalcnt[-5:])
        
        if prediction >= 1650:
            # 상승 추세 감지
            if recent_5_avg > current_totalcnt - 50:
                prediction += 30
            
            # 현재값이 높으면 추가 보정
            if current_totalcnt >= 1680:
                prediction += 20
        
        # 황금 패턴 감지
        golden_pattern = (seq_m14b[-1] > 300 and seq_m10a[-1] < 80)
        
        # 위험 구간 감지
        danger_in_seq = np.sum(seq_totalcnt >= 1700) > 0
def predict_latest():
    """
    가장 최근 280분 데이터로 15분 후 예측
    반환: {'prediction': int, 'status': str, 'prediction_time': str, 'danger_probability': int}
    에러 시: {'prediction': 0, 'status': 'Model operation failure/No data/Lack of data', ...}
    """
    
    # ⭐ 1. 모델 로드 (15분용 모델)
    model_file = 'model_13col_15.pkl'
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"❌ Model file error: {e}")
        return {
            'prediction': 0,
            'status': 'Model operation failure',
            'prediction_time': '',
            'danger_probability': 0,
            'error_message': f'No model file: {model_file}'
        }
    
    # ⭐ 2. CSV 파일 확인
    csv_file = 'data/2025_DATA.CSV'
    if not os.path.exists(csv_file):
        print(f"❌ No CSV file: {csv_file}")
        return {
            'prediction': 0,
            'status': 'No data',
            'prediction_time': '',
            'danger_probability': 0,
            'error_message': f'No data file: {csv_file}'
        }
    
    # ⭐ 3. 데이터 로드
    try:
        df = pd.read_csv(csv_file, on_bad_lines='skip', dtype={'CURRTIME': str})
    except Exception as e:
        print(f"❌ Data Load Error: {e}")
        return {
            'prediction': 0,
            'status': 'No data',
            'prediction_time': '',
            'danger_probability': 0,
            'error_message': f'Failed to load data: {e}'
        }
    
    # ⭐ 4. 데이터가 비어있는 경우
    if len(df) == 0:
        print(f"❌ No data: CSV file empty")
        return {
            'prediction': 0,
            'status': 'No data',
            'prediction_time': '',
            'danger_probability': 0,
            'error_message': 'CSV file is empty'
        }
    
    # ⭐ 5. 필수 컬럼 확인
    required_cols = [
        'M14AM14B', 'M14AM14BSUM', 'M14BM14A',
        'M14AM10A', 'M10AM14A', 'M16M14A', 'M14AM16SUM', 'TOTALCNT',
        'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED',
        'M14.QUE.OHT.OHTUTIL', 'M14.QUE.ALL.TRANSPORT4MINOVERCNT'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required column: {missing_cols}")
        return {
            'prediction': 0,
            'status': 'No data',
            'prediction_time': '',
            'danger_probability': 0,
            'error_message': f'Missing required column: {", ".join(missing_cols)}'
        }
    
    # ⭐ 6. 데이터 부족 체크
    if len(df) < 280:
        print(f"❌ Lack of data: {len(df)}EA (Minimum 280 EA required)")
        return {
            'prediction': 0,
            'status': 'Lack of data',
            'prediction_time': '',
            'danger_probability': 0,
            'error_message': f'Lack of data: {len(df)}EA (Minimum 280 EA required)'
        }
    
    # ⭐ 7. CURRTIME 파싱
    if 'CURRTIME' in df.columns:
        try:
            df['CURRTIME'] = df['CURRTIME'].astype(str).str.strip()
            df = df[df['CURRTIME'].str.len() == 12].copy()
            if len(df) == 0:
                raise ValueError("No valid CURRTIME")
            df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M', errors='coerce')
            df = df.dropna(subset=['CURRTIME']).copy()
            if len(df) < 280:
                raise ValueError(f"Lack of data after CURRTIME parsing: {len(df)}EA")
        except Exception as e:
            # CURRTIME 파싱 실패해도 가상 시간으로 진행
            base_time = datetime.now()
            df['CURRTIME'] = [base_time - timedelta(minutes=len(df)-1-i) for i in range(len(df))]
    else:
        base_time = datetime.now()
        df['CURRTIME'] = [base_time - timedelta(minutes=len(df)-1-i) for i in range(len(df))]
    
    # 최종 데이터 부족 재확인
    if len(df) < 280:
        print(f"❌ Lack of data: {len(df)}EA (Minimum 280 EA required)")
        return {
            'prediction': 0,
            'status': 'Lack of data',
            'prediction_time': '',
            'danger_probability': 0,
            'error_message': f'Lack of data: {len(df)}EA (Minimum 280 EA required)'
        }
    
    # ⭐ 8. Feature 추출 및 예측
    try:
        # 최근 280분 데이터 추출
        row_dict = {
            'M14AM14B': df['M14AM14B'].iloc[-280:].values,
            'M14AM14BSUM': df['M14AM14BSUM'].iloc[-280:].values,
            'M14BM14A': df['M14BM14A'].iloc[-280:].values,
            'M14AM10A': df['M14AM10A'].iloc[-280:].values,
            'M10AM14A': df['M10AM14A'].iloc[-280:].values,
            'M16M14A': df['M16M14A'].iloc[-280:].values,
            'M14AM16SUM': df['M14AM16SUM'].iloc[-280:].values,
            'TOTALCNT': df['TOTALCNT'].iloc[-280:].values,
            'M14.QUE.ALL.CURRENTQCREATED': df['M14.QUE.ALL.CURRENTQCREATED'].iloc[-280:].values,
            'M14.QUE.ALL.CURRENTQCOMPLETED': df['M14.QUE.ALL.CURRENTQCOMPLETED'].iloc[-280:].values,
            'M14.QUE.OHT.OHTUTIL': df['M14.QUE.OHT.OHTUTIL'].iloc[-280:].values,
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT': df['M14.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[-280:].values,
        }
        
        # 시간 정보 (⭐ 15분 후)
        current_time = df['CURRTIME'].iloc[-1]
        if pd.isna(current_time):
            current_time = datetime.now()
        prediction_time = current_time + timedelta(minutes=15)
        
        # 현재 상태
        seq_totalcnt = row_dict['TOTALCNT']
        seq_m14b = row_dict['M14AM14B']
        seq_m14b_sum = row_dict['M14AM14BSUM']
        seq_qc = row_dict['M14.QUE.ALL.CURRENTQCREATED']
        seq_qd = row_dict['M14.QUE.ALL.CURRENTQCOMPLETED']
        seq_gap = seq_qc - seq_qd
        seq_trans = row_dict['M14.QUE.ALL.TRANSPORT4MINOVERCNT']
        
        current_totalcnt = seq_totalcnt[-1]
        current_m14b = seq_m14b[-1]
        current_m14bsum = seq_m14b_sum[-1]
        current_gap = seq_gap[-1]
        current_trans = seq_trans[-1]
        
        # Feature 생성
        features = create_features_13col_optimized(row_dict)
        X_pred = pd.DataFrame([features])
        
        # 원본 예측
        pred_raw = model.predict(X_pred)[0]
        
        # boost 보정
        pred = adjust_light_plus(pred_raw, current_m14b, current_m14bsum, current_gap, current_trans)
        
        # 상태 판정
        pred_status = get_status_info(pred)
        
        # 패턴 감지
        gold_strict = (current_m14b > 520 and current_m14bsum > 588)
        gold_normal = (current_m14b > 517 and current_m14bsum > 576)
        danger_gap = current_gap > 300
        danger_trans = current_trans > 151
        
        # ⭐ 1700+ 위험 확률 계산
        danger_prob = 0
        
        if pred >= 1750:
            danger_prob = 100
        elif pred >= 1700:
            danger_prob = 95
        elif pred >= 1680:
            danger_prob = 75
        elif pred >= 1650:
            danger_prob = 50
        elif pred >= 1620:
            danger_prob = 30
        elif pred >= 1600:
            danger_prob = 15
        else:
            danger_prob = 5
        
        # 황금 패턴 보정
        if gold_strict:
            danger_prob = min(100, danger_prob + 20)
        elif gold_normal:
            danger_prob = min(100, danger_prob + 15)
        elif (current_m14b > 509 and current_m14bsum > 570):
            danger_prob = min(100, danger_prob + 10)
        
        # Gap/Trans 보정
        if danger_gap:
            danger_prob = min(100, danger_prob + (10 if current_gap > 350 else 5))
        if danger_trans:
            danger_prob = min(100, danger_prob + (10 if current_trans > 180 else 5))
        
        # 현재값 보정
        if current_totalcnt >= 1700:
            danger_prob = max(danger_prob, 85)
        elif current_totalcnt >= 1650:
            danger_prob = max(danger_prob, 60)
        
        danger_prob = max(0, min(100, danger_prob))
        
        # ⭐ 정상 결과 반환
        result = {
            'prediction': int(pred),
            'status': pred_status,
            'prediction_time': prediction_time.strftime('%Y-%m-%d %H:%M'),
            'danger_probability': danger_prob
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Predictive execution error: {e}")
        return {
            'prediction': 0,
            'status': 'Model operation failure',
            'prediction_time': '',
            'danger_probability': 0,
            'error_message': f'Prediction execution failed: {e}'
        }
def create_evaluation_sequences(self, df, numeric_cols):
    """평가용 시퀀스 생성"""
    X_list = []
    y_actual_list = []
    y_time_list = []
    physics_list = []
    timestamps = []
    
    data = df[numeric_cols].values
    target_idx = numeric_cols.index(self.target_col)
    
    # 사용 가능한 컬럼 확인
    available_inflow = [col for col in self.inflow_cols if col in numeric_cols]
    available_outflow = [col for col in self.outflow_cols if col in numeric_cols]
    
    print(f"\n📊 시퀀스 생성 중...")
    total_sequences = len(data) - self.seq_len - self.pred_len + 1
    
    # 시퀀스 생성
    for i in range(total_sequences):
        # 입력 시퀀스 (과거 20분: 00:00 ~ 00:19)
        X_seq = data[i:i+self.seq_len]
        
        # 실제 값 (미래 10분: 00:20 ~ 00:29)
        y_actual = data[i+self.seq_len:i+self.seq_len+self.pred_len, target_idx]
        
        # 타임스탬프 수정: 예측 시작 시점은 00:20
        start_time = df['timestamp'].iloc[i+self.seq_len]  # -1 제거
        pred_times = [df['timestamp'].iloc[i+self.seq_len+j] for j in range(self.pred_len)]
        
        # 물리 데이터 (마지막 입력 시점 = 00:19의 상태)
        current_state = data[i+self.seq_len-1]
        current_hubroom = current_state[target_idx]
        
        inflow_sum = sum([current_state[numeric_cols.index(col)] 
                        for col in available_inflow])
        outflow_sum = sum([current_state[numeric_cols.index(col)] 
                         for col in available_outflow])
        
        physics = np.array([current_hubroom, inflow_sum, outflow_sum])
        
        X_list.append(X_seq)
        y_actual_list.append(y_actual)
        y_time_list.append(pred_times)
        physics_list.append(physics)
        timestamps.append(start_time)  # 예측 시작 시점
    
    print(f"✅ 시퀀스 생성 완료: {len(X_list)}개")
    
    return (np.array(X_list), np.array(y_actual_list), 
            np.array(physics_list), timestamps, y_time_list)
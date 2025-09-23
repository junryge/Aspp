def create_transition_sequences(self, df):
        """전이 패턴 중심 시퀀스 생성 - [수정됨] 메모리 안정화를 위해 오버샘플링 비율 대폭 축소"""
        print("\n🔄 전이 패턴 시퀀스 생성 중... (메모리 안정화 버전)")
        
        X_data = df[self.feature_columns].values
        y_data = df['TOTALCNT'].shift(-self.pred_len).values
        
        X, y = [], []
        transition_X, transition_y = [], []  # 전이 패턴
        danger_X, danger_y = [], []          # 위험 패턴
        
        # 상승/하락 구분 카운터
        rise_signal_to_danger = 0
        rise_in_danger = 0
        fall_from_danger = 0
        
        for i in range(len(df) - self.seq_len - self.pred_len):
            seq_X = X_data[i:i+self.seq_len]
            target = y_data[i+self.seq_len-1]
            
            if np.isnan(target):
                continue
                
            current = df['TOTALCNT'].iloc[i+self.seq_len-1]
            
            X.append(seq_X)
            y.append(target)
            
            # ⭐ 상승/하락 구분
            is_rising = target > current  # 상승 중
            change_amount = target - current
            
            # 전이 패턴 탐지 및 오버샘플링 (복제 비율 대폭 축소)
            
            # 1. 신호→위험 상승 전이 (가장 중요!)
            if 1650 <= current <= 1699 and target >= 1700 and is_rising:
                # [수정] 20배 -> 5배
                for _ in range(5):
                    transition_X.append(seq_X)
                    transition_y.append(target)
                rise_signal_to_danger += 1
            
            # 2. 위험구간 내 상승 지속
            elif current >= 1700 and target >= 1700 and is_rising:
                # [수정] 15배 -> 4배
                for _ in range(4):
                    danger_X.append(seq_X)
                    danger_y.append(target)
                rise_in_danger += 1
            
            # 3. 위험구간에서 하락 (안정화)
            elif current >= 1700 and target < current:
                # [수정] 3배 -> 2배
                for _ in range(2):
                    transition_X.append(seq_X)
                    transition_y.append(target)
                fall_from_danger += 1
            
            # 4. 급상승 패턴 (50 이상)
            elif change_amount >= 50 and is_rising:
                # [수정] 12배 -> 3배
                for _ in range(3):
                    transition_X.append(seq_X)
                    transition_y.append(target)
            
            # 5. 신호구간 내 상승
            elif 1651 <= current <= 1699 and is_rising and change_amount >= 20:
                # [수정] 8배 -> 3배
                for _ in range(3):
                    transition_X.append(seq_X)
                    transition_y.append(target)
            
            # 6. 1680 이상에서 상승 (위험 임박)
            elif current >= 1680 and is_rising:
                # [수정] 10배 -> 3배
                for _ in range(3):
                    danger_X.append(seq_X)
                    danger_y.append(target)
        
        # 오버샘플링 데이터 추가
        if transition_X:
            X.extend(transition_X)
            y.extend(transition_y)
            print(f"  🔥 전이 패턴 {len(transition_X)}개 추가")
        
        if danger_X:
            X.extend(danger_X)
            y.extend(danger_y)
            print(f"  🔥 위험 패턴 {len(danger_X)}개 추가")
        
        print(f"\n  📊 패턴 통계:")
        print(f"    신호→위험 상승 전이: {rise_signal_to_danger}건")
        print(f"    위험구간 내 상승: {rise_in_danger}건")
        print(f"    위험구간에서 하락: {fall_from_danger}건")
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        print(f"  총 시퀀스: {X.shape[0]}개")
        
        # 셔플
        shuffle_idx = np.arange(len(X))
        np.random.shuffle(shuffle_idx)
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        return self.split_and_scale(X, y)
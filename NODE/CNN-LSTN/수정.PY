# 1. CheckpointManager 클래스의 save_scaler 메서드 수정
def save_scaler(self, scaler):
    """스케일러 저장 - 중요!"""
    joblib.dump(scaler, self.scaler_file)
    logger.info(f"스케일러 저장됨: {self.scaler_file}")  # ✓ 제거

# 2. CheckpointManager 클래스의 모든 로깅 메서드들 수정 (✓ 제거)
def save_state(self, state_dict):
    """현재 학습 상태 저장"""
    # JSON 직렬화를 위해 numpy 타입 변환
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # 재귀적으로 변환
    state_dict_converted = json.loads(
        json.dumps(state_dict, default=convert_numpy)
    )
    
    with open(self.state_file, 'w') as f:
        json.dump(state_dict_converted, f, indent=4)
    logger.info(f"학습 상태 저장됨: {self.state_file}")  # ✓ 제거

def load_state(self):
    """저장된 학습 상태 로드"""
    if os.path.exists(self.state_file):
        with open(self.state_file, 'r') as f:
            state = json.load(f)
        logger.info(f"학습 상태 로드됨: {self.state_file}")  # ✓ 제거
        return state
    return None

def save_data(self, data_dict):
    """전처리된 데이터 저장"""
    with open(self.data_file, 'wb') as f:
        pickle.dump(data_dict, f)
    logger.info(f"데이터 저장됨: {self.data_file}")  # ✓ 제거

def load_data(self):
    """저장된 데이터 로드"""
    if os.path.exists(self.data_file):
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"데이터 로드됨: {self.data_file}")  # ✓ 제거
        return data
    return None

def load_scaler(self):
    """저장된 스케일러 로드"""
    if os.path.exists(self.scaler_file):
        scaler = joblib.load(self.scaler_file)
        logger.info(f"스케일러 로드됨: {self.scaler_file}")  # ✓ 제거
        return scaler
    return None

def save_class_mapping(self, class_mapping):
    """클래스 매핑 정보 저장"""
    # int64를 str로 변환
    class_mapping_str = {str(k): int(v) for k, v in class_mapping.items()}
    with open(self.class_mapping_file, 'w') as f:
        json.dump(class_mapping_str, f, indent=4)
    logger.info(f"클래스 매핑 저장됨: {self.class_mapping_file}")  # ✓ 제거

def load_class_mapping(self):
    """클래스 매핑 정보 로드"""
    if os.path.exists(self.class_mapping_file):
        with open(self.class_mapping_file, 'r') as f:
            mapping = json.load(f)
        # 필요시 다시 int로 변환
        mapping_int = {int(k): v for k, v in mapping.items()}
        logger.info(f"클래스 매핑 로드됨: {self.class_mapping_file}")  # ✓ 제거
        return mapping_int
    return None

def save_history(self, history):
    """학습 히스토리 저장"""
    with open(self.history_file, 'wb') as f:
        pickle.dump(history, f)
    logger.info(f"학습 히스토리 저장됨: {self.history_file}")  # ✓ 제거

def load_history(self):
    """학습 히스토리 로드"""
    if os.path.exists(self.history_file):
        with open(self.history_file, 'rb') as f:
            history = pickle.load(f)
        logger.info(f"학습 히스토리 로드됨: {self.history_file}")  # ✓ 제거
        return history
    return None

# 3. PeriodicCheckpoint 클래스의 on_epoch_end 메서드 수정
def on_epoch_end(self, epoch, logs=None):
    # 히스토리 업데이트
    for key in self.history.keys():
        if logs and key in logs:
            self.history[key].append(logs[key])
    
    # 주기적 저장
    if (epoch + 1) % self.save_freq == 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"에폭 {epoch + 1}: 체크포인트 저장 중...")
        
        # 1. 모델 저장
        model_path = self.checkpoint_manager.save_model_checkpoint(self.model, epoch + 1)
        
        # 2. 스케일러 저장 (중요!)
        if self.scaler is not None:
            self.checkpoint_manager.save_scaler(self.scaler)
        
        # 3. 상태 저장
        state = {
            'current_epoch': epoch + 1,
            'model_path': model_path,
            'best_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else float('inf'),
            'training_completed': False,
            'last_save_time': datetime.now().isoformat()
        }
        self.checkpoint_manager.save_state(state)
        
        # 4. 히스토리 저장
        self.checkpoint_manager.save_history(self.history)
        
        logger.info(f"체크포인트 저장 완료 (에폭 {epoch + 1})")  # ✓ 제거
        logger.info(f"{'='*60}\n")
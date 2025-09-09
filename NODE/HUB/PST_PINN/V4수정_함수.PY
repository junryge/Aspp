# ========================================
# V4 극단값 손실 함수 (False Positive 방지) - 수정 완료
# ========================================

class ExtremeLossV4(keras.losses.Loss):
    def __init__(self, extreme_focus=True):
        super().__init__()
        self.extreme_focus = extreme_focus
        
    def call(self, y_true, y_pred):
        # ⭐ 추가: shape 맞추기
        y_pred = tf.reshape(y_pred, tf.shape(y_true))
        
        error = y_true - y_pred
        mae = tf.abs(error)
        
        if self.extreme_focus:
            # V4: 335+ 극단값에 더 강한 페널티
            weights = tf.where(y_true >= 0.9, 25.0,    # 335+ (정규화 기준)
                     tf.where(y_true >= 0.8, 15.0,     # 310-335
                     tf.where(y_true >= 0.7, 8.0,      # 300-310
                     tf.where(y_true >= 0.5, 3.0,      # 250-300
                              1.0))))                    # 나머지
            
            # 극단값 미탐지 추가 페널티 (False Negative)
            miss_penalty = tf.where(
                tf.logical_and(y_true >= 0.9, y_pred < 0.85),
                30.0 * mae,  # 335+ 미탐지 시 강력한 페널티
                0.0
            )
            
            # False Positive 방지 페널티 (300 이하를 300 이상으로 예측)
            false_positive_penalty = tf.where(
                tf.logical_and(y_true < 0.7, y_pred >= 0.7),  # 300 이하를 300+ 예측
                20.0 * mae,  # 오탐 페널티
                0.0
            )
            
            # 특히 200-250 구간을 300+로 예측하면 더 강한 페널티
            severe_fp_penalty = tf.where(
                tf.logical_and(y_true < 0.5, y_pred >= 0.7),  # 250 이하를 300+ 예측
                35.0 * mae,  # 심각한 오탐 페널티
                0.0
            )
            
            # ⭐ 수정: reduce_mean 제거
            loss = weights * mae + miss_penalty + false_positive_penalty + severe_fp_penalty
            return loss  # 그대로 반환
            
        else:
            # Model 1: 균형잡힌 가중치 + 경미한 오탐 방지
            weights = tf.where(y_true >= 0.7, 5.0,
                     tf.where(y_true >= 0.5, 2.0, 1.0))
            
            # Model 1에도 경미한 오탐 방지
            fp_penalty = tf.where(
                tf.logical_and(y_true < 0.6, y_pred >= 0.7),
                5.0 * mae,
                0.0
            )
            
            # ⭐ 수정: reduce_mean 제거
            loss = weights * mae + fp_penalty
            return loss  # 그대로 반환




def main():
    # 체크포인트 관리자
    ckpt = CheckpointManager()
    
    # 이전 상태 확인
    state = ckpt.load_state()
    
    if state:
        print(f"\n📂 이전 학습 상태 발견! (Step {state.get('step', 1)})")
        
        resume = input("이어서 진행하시겠습니까? (y: 이어서, n: 처음부터): ").lower()
        
        if resume != 'y':
            ckpt.clear_state()
            state = {}  # ⭐ 수정: None → {}
            step = 1
        else:
            step = state.get('step', 1)
            print(f"✅ Step {step}부터 재개합니다.")
    else:
        state = {}
        step = 1
        print("\n이전 학습 상태가 없습니다.")
        input("Enter를 눌러 시작하세요...")
    
    # 데이터 처리기
    processor = DataProcessorV4()
    
    # 나머지 코드는 그대로...
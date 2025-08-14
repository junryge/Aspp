"""
TensorBoard 실시간 모니터링 설정
================================
"""

# ===================================
# 1. TensorBoard 설치
# ===================================
# pip install tensorboard

# ===================================
# 2. 학습 코드에 추가 (train_model_improved 함수 수정)
# ===================================

from tensorflow.keras.callbacks import TensorBoard
import datetime

def train_model_improved_with_tensorboard(model, model_name, X_train, y_train, X_val, y_val, 
                         epochs, batch_size, checkpoint_manager):
    """TensorBoard 추가된 학습 함수"""
    
    # TensorBoard 로그 디렉토리
    log_dir = f"logs/fit/{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # 콜백 설정
    callbacks = [
        # TensorBoard 추가!
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,  # 매 에폭마다 히스토그램
            write_graph=True,   # 모델 구조 시각화
            write_images=True,  # 가중치 이미지
            update_freq='epoch',
            profile_batch=0     # 프로파일링 비활성화 (빠르게)
        ),
        
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
        ModelCheckpoint(f'{checkpoint_manager.checkpoint_dir}/{model_name}_best.h5', 
                       save_best_only=True, monitor='val_loss')
    ]
    
    # 학습
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# ===================================
# 3. 실시간 모니터링 (별도 스크립트)
# ===================================

import json
import time
import os

def monitor_training_progress():
    """
    체크포인트 파일을 읽어서 진행상황 모니터링
    TensorBoard 없이도 가능!
    """
    
    checkpoint_file = 'checkpoints_v3/training_state.json'
    
    print("=" * 60)
    print("📊 학습 진행 상황 모니터링")
    print("=" * 60)
    
    while True:
        try:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    state = json.load(f)
                
                current_model = state.get('current_model', 'Unknown')
                current_epoch = state.get('current_epoch', 0)
                
                # 모델별 이력 확인
                if 'model_histories' in state:
                    for model_name, history in state['model_histories'].items():
                        if 'val_loss' in history and len(history['val_loss']) > 0:
                            latest_val_loss = history['val_loss'][-1]
                            latest_val_mae = history.get('val_mae', [0])[-1]
                            
                            print(f"\n모델: {model_name}")
                            print(f"  Epoch: {len(history['val_loss'])}")
                            print(f"  Val Loss: {latest_val_loss:.4f}")
                            print(f"  Val MAE: {latest_val_mae:.4f}")
                            print(f"  실제 MAE: {latest_val_mae * 200:.1f}")
                            
                            # 성능 판단
                            if latest_val_mae * 200 < 60:
                                print("  ✅ 목표 달성!")
                            elif latest_val_mae * 200 < 80:
                                print("  🔶 양호")
                            else:
                                print("  ⏳ 개선 중...")
                
                print("\n" + "-" * 40)
                print(f"현재 시각: {datetime.datetime.now().strftime('%H:%M:%S')}")
                print("10초 후 업데이트...")
                
            else:
                print("체크포인트 파일이 없습니다. 기다리는 중...")
            
            time.sleep(10)  # 10초마다 업데이트
            
        except KeyboardInterrupt:
            print("\n모니터링 종료")
            break
        except Exception as e:
            print(f"오류: {e}")
            time.sleep(5)

# ===================================
# 4. 간단한 진행 상황 확인 (더 간단!)
# ===================================

def quick_check():
    """
    현재 상태만 빠르게 확인
    """
    import glob
    import matplotlib.pyplot as plt
    
    # 가장 최근 체크포인트 찾기
    checkpoints = glob.glob('checkpoints_v3/*.h5')
    
    if checkpoints:
        latest = max(checkpoints, key=os.path.getctime)
        print(f"최신 체크포인트: {latest}")
        print(f"수정 시간: {datetime.datetime.fromtimestamp(os.path.getmtime(latest))}")
    
    # training.log 파일 확인
    if os.path.exists('training_improved.log'):
        with open('training_improved.log', 'r') as f:
            lines = f.readlines()
            
        # 최근 10줄 출력
        print("\n최근 로그:")
        for line in lines[-10:]:
            if 'Epoch' in line or 'loss' in line or 'MAE' in line:
                print(line.strip())
    
    # 간단한 그래프
    try:
        with open('checkpoints_v3/training_state.json', 'r') as f:
            state = json.load(f)
        
        if 'model_histories' in state:
            for model_name, history in state['model_histories'].items():
                if 'val_loss' in history:
                    plt.figure(figsize=(10, 4))
                    
                    plt.subplot(1, 2, 1)
                    plt.plot(history.get('loss', []), label='Train Loss')
                    plt.plot(history.get('val_loss', []), label='Val Loss')
                    plt.title(f'{model_name} - Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(True)
                    
                    plt.subplot(1, 2, 2)
                    val_mae = history.get('val_mae', [])
                    real_mae = [m * 200 for m in val_mae]
                    plt.plot(real_mae)
                    plt.axhline(y=60, color='r', linestyle='--', label='목표 (60)')
                    plt.title(f'{model_name} - 실제 MAE')
                    plt.xlabel('Epoch')
                    plt.ylabel('MAE')
                    plt.legend()
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(f'progress_{model_name}.png')
                    plt.show()
                    
                    print(f"\n그래프 저장: progress_{model_name}.png")
    except:
        pass

# ===================================
# 5. 실행 방법
# ===================================

if __name__ == "__main__":
    print("""
    선택하세요:
    1. TensorBoard 실행 (설치 필요)
    2. 실시간 모니터링 (TensorBoard 없이)
    3. 빠른 상태 확인
    """)
    
    choice = input("선택 (1/2/3): ")
    
    if choice == "1":
        os.system("tensorboard --logdir=logs/fit")
        print("브라우저에서 http://localhost:6006 접속!")
        
    elif choice == "2":
        monitor_training_progress()
        
    elif choice == "3":
        quick_check()
import tensorflow as tf
import time
from datetime import datetime

# GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 growth 비활성화 (전체 메모리 할당)
        tf.config.experimental.set_memory_growth(gpus[0], False)
        # 60GB 할당
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=60*1024)]
        )
    except RuntimeError as e:
        print(e)

print(f"TensorFlow: {tf.__version__}")
print(f"GPU 개수: {len(gpus)}")

if not gpus:
    print("GPU 없음!")
    exit()

print("\n60GB 메모리 할당 중...")

with tf.device('/GPU:0'):
    # 15GB 텐서 4개 = 60GB
    tensors = []
    for i in range(4):
        size = int(15 * 1024**3 / 4)  # 15GB (float32 = 4bytes)
        tensor = tf.random.normal((size,))
        tensors.append(tensor)
        print(f"텐서 {i+1}/4 할당 완료 (15GB)")

print("\nGPU 연산 시작... (Ctrl+C로 종료)")

iteration = 0
start_time = time.time()

try:
    while True:
        iteration += 1
        
        with tf.device('/GPU:0'):
            for idx, tensor in enumerate(tensors):
                result = tf.sin(tensor) * tf.cos(tensor)
                result = tf.reduce_mean(result)
        
        if iteration % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"반복: {iteration} | 경과시간: {elapsed:.1f}초")
        
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\n테스트 종료!")
    print(f"총 반복: {iteration}")
    print(f"총 시간: {time.time() - start_time:.1f}초")
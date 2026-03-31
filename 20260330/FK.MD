# Enhanced LSTM 오류 분석 및 K8s GPU 할당 가이드

> **2026년 3월 31일 09시 08분 오류 보고서**

---

## 1. 오류 요약

| 항목 | 내용 |
|------|------|
| Pod | `amhs-ai-job-predict-jupyternb-7c745f9f8f-5pkkl` |
| Namespace | `amhs-ai` |
| 환경 | `prd` (프로덕션) |
| 위치 | `hy-koic-k8s-day-p01` |
| 알림 발생 시각 | 2026-03-31 00:07:57 UTC (KST 09:07:57) |

### 1-1. K8s 알림: PODCPUUsageExceed

```
alertname: PODCPUUsageExceed
category1: Pod
category2: CPU
category3: ThresholdExceed
figure: 97.99978016936006 (%)
crtcallval: 90 (%)
measdurval: 2m
measobsrvval: 5m
```

- CPU 사용률이 **97.99%** 로 임계값 90%를 초과하여 경고 발생
- 측정 조건: 5분간 관측, 2분 지속 시 알림 발동
- Pod 클래스: `kubepods-burstable` (리소스 제한이 있는 burstable QoS)

### 1-2. LSTM 학습 오류: 메모리 할당 실패

```
2025-08-12 20:18:07,000 - INFO  - enhanced_lstm - Epoch 8/300
2025-08-12 20:18:07,015 - ERROR - enhanced_lstm 모델 학습 중 오류 발생:
Unable to allocate 3.81 GiB for an array with shape (210343, 30, 81) and data type float64
```

### 1-3. 학습 상태 (Epoch 7까지 정상 진행)

| 항목 | Train | Validation |
|------|-------|------------|
| Loss | - | 0.9090 |
| Regression MAE | - | 0.5770 |
| Spike Accuracy | 0.9791 | 0.8479 |
| Spike Precision | 0.9041 | 0.1329 |
| Spike Recall | 0.9970 | 0.9763 |
| Learning Rate | 3.0e-04 | - |

---

## 2. 원인 분석

### 2-1. 두 오류의 연관관계

두 오류는 동일한 Pod에서 동시에 발생한 연쇄 장애다.

```
Enhanced LSTM Epoch 8 시작
    ↓
NumPy sliding window 배열 생성 시도 (CPU에서 실행)
    ↓
210,343 × 30 × 81 × 8 bytes (float64) = 3.81 GiB 할당 시도
    ↓
CPU 사용률 97.99% 도달 → K8s PODCPUUsageExceed 경고 발동
    ↓
시스템 RAM 부족으로 배열 할당 실패 → 학습 중단
```

### 2-2. 근본 원인: Pod에 GPU가 할당되지 않음

V100 GPU 3장이 클러스터에 존재하지만, 해당 Jupyter Pod 스펙에 `nvidia.com/gpu` 리소스 요청이 없어 **GPU 없이 CPU만으로 학습이 수행**되고 있다.

K8s에서 GPU는 CPU/메모리와 달리 자동 할당되지 않으며, Pod 스펙에 명시적으로 요청해야만 사용 가능하다.

그 결과:

- 모든 연산이 CPU에서 수행 → CPU 과부하
- NumPy 배열 생성이 시스템 RAM에서 처리 → burstable Pod의 제한된 메모리 내에서 3.81GB 할당 실패
- GPU 메모리(V100 32GB × 3 = 96GB)는 전혀 활용되지 않음

### 2-3. Epoch 8에서 터진 이유

Epoch 1~7은 정상 진행되었으므로 다음 가능성이 있다:

- 콜백이나 augmentation 로직에서 매 epoch마다 배열을 새로 생성하면서, 이전 배열이 GC되기 전에 새 배열 할당 시도
- Epoch 반복 중 중간 결과가 해제되지 않아 가용 RAM이 점차 감소
- 다른 프로세스의 메모리 점유 증가로 가용 공간 부족

### 2-4. 추가 관찰: 모델 성능 이슈

- Val Spike Precision **0.1329**로 매우 낮음 → False Positive 과다 발생
- Train Spike Recall 0.9970 vs Val 0.9763 → 과적합 경향

---

## 3. 즉시 조치: 메모리 오류 해결

GPU 할당과 별개로, 현재 메모리 오류를 먼저 해결한다.

### 3-1. float32 변환 (메모리 50% 절감)

LSTM 학습에 float64는 불필요하다. 3.81GB → ~1.9GB로 즉시 절감.

```python
X = X.astype(np.float32)
y = y.astype(np.float32)
```

### 3-2. tf.data.Dataset Generator 방식 (근본 해결)

전체 배열을 메모리에 올리지 않고 배치 단위로 생성하여 전달한다.

```python
def data_generator(df, seq_len=30, batch_size=256):
    while True:
        for start in range(0, len(df) - seq_len, batch_size):
            batch_x, batch_y_reg, batch_y_spike = [], [], []
            for i in range(start, min(start + batch_size, len(df) - seq_len)):
                batch_x.append(df[i:i+seq_len, :-2])
                batch_y_reg.append(df[i+seq_len, -2])
                batch_y_spike.append(df[i+seq_len, -1])
            yield (
                np.array(batch_x, dtype=np.float32),
                {
                    'regression_output': np.array(batch_y_reg, dtype=np.float32),
                    'spike_output': np.array(batch_y_spike, dtype=np.float32)
                }
            )

dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(data_array, seq_len=30, batch_size=256),
    output_signature=(
        tf.TensorSpec(shape=(None, 30, 81), dtype=tf.float32),
        {
            'regression_output': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'spike_output': tf.TensorSpec(shape=(None,), dtype=tf.float32)
        }
    )
).prefetch(tf.data.AUTOTUNE)

model.fit(dataset, steps_per_epoch=len(df) // 256, epochs=300)
```

---

## 4. GPU 할당 방법

### 4-1. 현재 Pod 상태 확인

```bash
# Pod GPU 할당 여부 확인
kubectl describe pod amhs-ai-job-predict-jupyternb-7c745f9f8f-5pkkl \
  -n amhs-ai | grep -A5 "Limits\|Requests\|nvidia"

# 클러스터 내 GPU 노드 확인
kubectl get nodes \
  -o custom-columns="NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

# GPU 노드 라벨 확인
kubectl get nodes --show-labels | grep gpu
```

### 4-2. GPU Pod 스펙 작성법

K8s에서 GPU를 사용하려면 `resources.limits`에 `nvidia.com/gpu`를 명시해야 한다.

#### GPU 1장 사용

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: amhs-ai-lstm-gpu
  namespace: amhs-ai
spec:
  containers:
  - name: lstm-training
    image: <TF-GPU 이미지>
    resources:
      limits:
        nvidia.com/gpu: 1
        memory: "32Gi"
        cpu: "8"
      requests:
        memory: "16Gi"
        cpu: "4"
    command: ["python", "enhanced_lstm.py"]
  restartPolicy: OnFailure
```

#### GPU 3장 사용 (V100 전체)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: amhs-ai-lstm-multigpu
  namespace: amhs-ai
spec:
  containers:
  - name: lstm-training
    image: <TF-GPU 이미지>
    resources:
      limits:
        nvidia.com/gpu: 3
        memory: "96Gi"
        cpu: "16"
      requests:
        memory: "64Gi"
        cpu: "8"
    command: ["python", "enhanced_lstm.py"]
  restartPolicy: OnFailure
```

#### GPU 리소스 규칙

- GPU는 `limits`에만 지정해도 된다 (K8s가 자동으로 requests = limits 설정)
- `limits`와 `requests` 둘 다 지정하면 반드시 같은 값이어야 한다
- GPU는 정수 단위로만 요청 가능 (0.5 불가)

### 4-3. 기존 Jupyter Deployment에 GPU 추가

```bash
# 현재 Deployment 스펙 확인
kubectl get deployment amhs-ai-job-predict-jupyternb -n amhs-ai -o yaml > jupyter_deploy.yaml
```

`jupyter_deploy.yaml`에서 수정할 부분:

```yaml
spec:
  template:
    spec:
      containers:
      - name: jupyternb
        resources:
          limits:
            nvidia.com/gpu: 1    # ← 추가
            memory: "32Gi"
            cpu: "8"
          requests:
            memory: "16Gi"
            cpu: "4"
```

```bash
# 수정 적용
kubectl apply -f jupyter_deploy.yaml
```

또는 직접 편집:

```bash
kubectl edit deployment amhs-ai-job-predict-jupyternb -n amhs-ai
```

### 4-4. GPU 할당 후 컨테이너 내부 확인

```bash
# Pod 접속
kubectl exec -it <pod-name> -n amhs-ai -- bash

# GPU 확인
nvidia-smi

# TensorFlow GPU 인식 확인
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# CUDA 버전 확인
nvcc --version
```

---

## 5. Multi-GPU 학습 코드 (TensorFlow)

Pod에 GPU가 할당된 후, TensorFlow에서 Multi-GPU를 활용하는 방법이다.

```python
import tensorflow as tf

# GPU 확인
gpus = tf.config.list_physical_devices('GPU')
print(f"사용 가능한 GPU: {len(gpus)}개")

# GPU 메모리 증가 허용 (OOM 방지)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Multi-GPU 전략
strategy = tf.distribute.MirroredStrategy()
print(f"사용 GPU 수: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = build_model()
    model.compile(
        optimizer='adam',
        loss={
            'regression_output': 'mse',
            'spike_output': 'binary_crossentropy'
        },
        metrics={
            'regression_output': ['mae'],
            'spike_output': ['accuracy',
                             tf.keras.metrics.Precision(name='precision'),
                             tf.keras.metrics.Recall(name='recall')]
        }
    )

# 학습 (GPU 수에 맞춰 batch_size 스케일링)
BATCH_SIZE_PER_GPU = 256
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_GPU * strategy.num_replicas_in_sync

model.fit(dataset, steps_per_epoch=len(df) // GLOBAL_BATCH_SIZE, epochs=300)
```

---

## 6. 조치 체크리스트

### 즉시 조치

- [ ] `kubectl describe pod` → GPU 할당 여부 확인
- [ ] `free -h` → Pod 시스템 RAM 가용량 확인
- [ ] float64 → float32 변환 적용
- [ ] Generator 방식 전환 (전체 배열 일괄 생성 제거)

### GPU 할당 조치

- [ ] 클러스터 GPU 노드 존재 여부 확인 (`kubectl get nodes`)
- [ ] NVIDIA device plugin DaemonSet 동작 확인
- [ ] Deployment 스펙에 `nvidia.com/gpu` 추가
- [ ] Pod 재시작 후 `nvidia-smi` / `tf.config.list_physical_devices('GPU')` 확인
- [ ] MirroredStrategy 적용 (멀티GPU 시)

### 모델 성능 개선 (후속)

- [ ] Val Spike Precision 0.1329 → threshold 조정 또는 모델 구조 변경
- [ ] Train vs Val 격차 → 과적합 대응 (dropout, regularization, 데이터 증강)

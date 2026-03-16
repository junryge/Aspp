# React에서 `ALL.glb` / `ALL_cpu_fast_hq.glb` 사용 가이드 (한국어)

이 문서는 React(Three.js / React Three Fiber)에서 모델을 안정적으로 로드하고,
요청하신 기준인 **CPU 로딩 속도 개선** + **GPU 최적 조건**을 적용하는 방법을 정리합니다.

## 1) 어떤 파일을 쓰면 좋은가

- `ALL.glb`
  - 원본 모델
  - Draco 압축 사용
- `ALL_cpu_fast_hq.glb`
  - 최적화 버전(권장)
  - 실제 서비스/테스트에서 먼저 써보는 것을 추천

권장 순서:
1. 기본은 `ALL_cpu_fast_hq.glb`
2. 품질 비교가 필요할 때만 `ALL.glb` 로드

---

## 2) GPU 관련 핵심 사실

- 브라우저 3D 렌더링(WebGL)은 기본적으로 GPU 사용이 필요합니다.
- CPU는 파일 읽기/파싱/JS 실행을 담당하고,
- 최종 렌더링은 GPU가 담당합니다.

즉, "GPU 없이 동일한 인터랙티브 3D 품질"은 현실적으로 어렵습니다.

---

## 3) 패키지 설치

```bash
npm install three @react-three/fiber @react-three/drei
```

---

## 4) 파일 배치

프로젝트 기준:

- `public/models/ALL.glb`
- `public/models/ALL_cpu_fast_hq.glb`

---

## 5) WebGL 가능 여부 체크

`src/utils/webgl.ts`

```ts
export function hasWebGL(): boolean {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    return !!gl;
  } catch {
    return false;
  }
}
```

---

## 6) 모델 로더 (Draco + Meshopt)

`src/components/Model.tsx`

```tsx
import { useMemo } from 'react';
import { useLoader } from '@react-three/fiber';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';

type Props = { url: string };

export default function Model({ url }: Props) {
  const gltf = useLoader(GLTFLoader, url, (loader) => {
    const draco = new DRACOLoader();
    draco.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');
    loader.setDRACOLoader(draco);
    loader.setMeshoptDecoder(MeshoptDecoder);
  });

  const scene = useMemo(() => gltf.scene.clone(), [gltf.scene]);
  return <primitive object={scene} />;
}
```

---

## 7) React 예시 (GPU 최적 기본값 포함)

`src/App.tsx`

```tsx
import { Suspense, useMemo, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Html, useProgress } from '@react-three/drei';
import Model from './components/Model';
import { hasWebGL } from './utils/webgl';

function Loader() {
  const { progress } = useProgress();
  return <Html center>{progress.toFixed(0)}%</Html>;
}

export default function App() {
  const [useHQ, setUseHQ] = useState(true);
  const webglOK = useMemo(() => hasWebGL(), []);

  const modelUrl = useHQ
    ? '/models/ALL_cpu_fast_hq.glb'
    : '/models/ALL.glb';

  if (!webglOK) {
    return <div style={{ padding: 24 }}>WebGL/GPU를 사용할 수 없는 환경입니다.</div>;
  }

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <div style={{ position: 'fixed', top: 12, left: 12, zIndex: 10 }}>
        <button onClick={() => setUseHQ(true)}>최적화 모델</button>
        <button onClick={() => setUseHQ(false)} style={{ marginLeft: 8 }}>원본 모델</button>
      </div>

      <Canvas
        camera={{ position: [5, 3, 5], fov: 45 }}
        dpr={[1, 1.5]}
        gl={{ antialias: true, powerPreference: 'high-performance' }}
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 10, 5]} intensity={1.2} castShadow={false} />

        <Suspense fallback={<Loader />}>
          <Model url={modelUrl} />
        </Suspense>

        <OrbitControls makeDefault enableDamping />
      </Canvas>
    </div>
  );
}
```

---

## 8) CPU 로딩 속도 빠르게 하는 설정 (중요)

1. 모델 선택
- 기본은 `ALL_cpu_fast_hq.glb` 사용

2. 초기 로딩 전략
- 첫 진입 시 가벼운 로딩 UI(Suspense) 표시
- 필요 시 route-level lazy loading 적용

3. 네트워크/파일
- 정적 파일 캐시 헤더(`Cache-Control`) 적극 사용
- CDN 사용 가능하면 적용

4. 디코더 설정
- Draco, Meshopt 디코더를 정확히 연결
- 디코더 누락 시 로딩 실패 또는 지연 발생

5. 불필요한 동시 작업 줄이기
- 모델 로딩 중 무거운 API 요청/대량 상태 업데이트 최소화

6. 반복 로딩 방지
- 같은 URL을 계속 바꾸지 않기
- 모델 재마운트 최소화 (`useMemo`, `Suspense` 활용)

---

## 9) GPU 최적 조건 (프레임 안정화)

1. DPR 제한
- `dpr={[1, 1.5]}` 또는 저사양은 `[1, 1]`

2. 그림자 최소화
- `castShadow`/`shadowMap`은 필요할 때만

3. 후처리 효과 최소화
- Bloom/DOF/SSAO 같은 포스트이펙트는 신중히 사용

4. 광원 수 줄이기
- 방향광/환경광 위주 단순 구성

5. 렌더러 옵션
- `powerPreference: 'high-performance'`

6. 카메라/컨트롤 안정화
- 과한 auto-rotate, 과한 damping 값 지양

---

## 10) 자주 나는 문제

### 모델이 로드되지 않음
확인 목록:
1. 경로가 `/models/...` 형식인지
2. `setMeshoptDecoder(MeshoptDecoder)` 적용했는지
3. Draco decoder path 접속 가능한지
4. 브라우저 콘솔에 CORS/404 에러가 없는지

### 화면은 뜨는데 너무 느림
1. `ALL_cpu_fast_hq.glb` 사용 여부 확인
2. DPR 낮추기 (`[1,1]`)
3. 그림자 끄기
4. 라이트/이펙트 줄이기

---

원하면 다음 단계로, 이 가이드 기준의 **실행 가능한 React 템플릿 파일 세트**(`App.tsx`, `Model.tsx`, `webgl.ts`)를 바로 생성해드릴 수 있습니다.

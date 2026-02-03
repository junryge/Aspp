"""
박스 위치 이상 탐지 (Box Position Anomaly Detection)
- 방법1: OpenCV 템플릿 매칭 (간단, CPU)
- 방법2: PatchCore (Anomalib, 정확)

pip install opencv-python numpy anomalib torch torchvision
"""

import cv2
import numpy as np
import os
from pathlib import Path

# ============================================================
# 방법 1: OpenCV 기반 (간단하고 빠름, CPU OK)
# ============================================================

class BoxPositionDetector:
    """OpenCV 기반 박스 위치 이상 탐지"""
    
    def __init__(self, reference_image_path):
        """
        Args:
            reference_image_path: 정상 박스 이미지 경로
        """
        self.reference = cv2.imread(reference_image_path)
        if self.reference is None:
            raise ValueError(f"Cannot load reference image: {reference_image_path}")
        
        self.ref_gray = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY)
        self.ref_center = self._find_box_center(self.ref_gray)
        self.ref_angle = self._find_box_angle(self.ref_gray)
        
        print(f"Reference - Center: {self.ref_center}, Angle: {self.ref_angle:.1f}°")
    
    def _find_box_center(self, gray_img):
        """박스 중심점 찾기"""
        # 이진화
        _, thresh = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 가장 큰 컨투어 (박스)
        largest = max(contours, key=cv2.contourArea)
        
        # 중심점 계산
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return (cx, cy)
    
    def _find_box_angle(self, gray_img):
        """박스 회전 각도 찾기"""
        _, thresh = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
        
        largest = max(contours, key=cv2.contourArea)
        
        # 최소 외접 사각형
        rect = cv2.minAreaRect(largest)
        angle = rect[2]
        
        # 각도 정규화 (-45 ~ 45)
        if angle < -45:
            angle += 90
        
        return angle
    
    def detect(self, test_image_path, position_threshold=30, angle_threshold=10):
        """
        이상 탐지 수행
        
        Args:
            test_image_path: 테스트 이미지 경로
            position_threshold: 위치 허용 오차 (픽셀)
            angle_threshold: 각도 허용 오차 (도)
        
        Returns:
            dict: {
                'is_anomaly': bool,
                'position_shift': float,
                'angle_shift': float,
                'anomaly_type': str,
                'center': tuple,
                'angle': float
            }
        """
        test_img = cv2.imread(test_image_path)
        if test_img is None:
            raise ValueError(f"Cannot load test image: {test_image_path}")
        
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        
        # 박스 위치/각도 찾기
        test_center = self._find_box_center(test_gray)
        test_angle = self._find_box_angle(test_gray)
        
        if test_center is None or self.ref_center is None:
            return {
                'is_anomaly': True,
                'anomaly_type': 'BOX_NOT_FOUND',
                'position_shift': -1,
                'angle_shift': -1
            }
        
        # 위치 차이 계산
        position_shift = np.sqrt(
            (test_center[0] - self.ref_center[0])**2 + 
            (test_center[1] - self.ref_center[1])**2
        )
        
        # 각도 차이 계산
        angle_shift = abs(test_angle - self.ref_angle)
        
        # 이상 판정
        is_position_anomaly = position_shift > position_threshold
        is_angle_anomaly = angle_shift > angle_threshold
        is_anomaly = is_position_anomaly or is_angle_anomaly
        
        # 이상 유형 결정
        if is_position_anomaly and is_angle_anomaly:
            anomaly_type = 'SHIFTED_AND_ROTATED'
        elif is_position_anomaly:
            anomaly_type = 'SHIFTED'
        elif is_angle_anomaly:
            anomaly_type = 'ROTATED'
        else:
            anomaly_type = 'NORMAL'
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_type': anomaly_type,
            'position_shift': round(position_shift, 2),
            'angle_shift': round(angle_shift, 2),
            'center': test_center,
            'angle': round(test_angle, 2)
        }
    
    def visualize(self, test_image_path, output_path=None):
        """결과 시각화"""
        result = self.detect(test_image_path)
        
        img = cv2.imread(test_image_path)
        
        # 정상 위치 표시 (녹색)
        if self.ref_center:
            cv2.circle(img, self.ref_center, 10, (0, 255, 0), 2)
            cv2.putText(img, "REF", (self.ref_center[0]-15, self.ref_center[1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 현재 위치 표시
        if result['center']:
            color = (0, 0, 255) if result['is_anomaly'] else (0, 255, 0)
            cv2.circle(img, result['center'], 10, color, -1)
        
        # 결과 텍스트
        status = "ANOMALY" if result['is_anomaly'] else "NORMAL"
        color = (0, 0, 255) if result['is_anomaly'] else (0, 255, 0)
        
        cv2.putText(img, f"{status}: {result['anomaly_type']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, f"Shift: {result['position_shift']}px", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Angle: {result['angle_shift']} deg", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"Saved: {output_path}")
        
        return img, result


# ============================================================
# 방법 2: Anomalib PatchCore (더 정확, GPU 권장)
# ============================================================

def train_patchcore_anomalib(data_dir, category="box"):
    """
    Anomalib PatchCore 학습
    
    폴더 구조:
    data_dir/
        train/
            good/
                normal_001.png
                normal_002.png
                ...
        test/
            good/
                test_normal_001.png
            anomaly/
                test_anomaly_001.png
    """
    try:
        from anomalib.data import Folder
        from anomalib.models import Patchcore
        from anomalib.engine import Engine
        from anomalib.callbacks import ModelCheckpoint
    except ImportError:
        print("Anomalib not installed. Run: pip install anomalib")
        return None
    
    # 데이터 모듈
    datamodule = Folder(
        root=data_dir,
        normal_dir="train/good",
        abnormal_dir="test/anomaly",
        normal_test_dir="test/good",
        image_size=(256, 256),
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
    )
    
    # 모델
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
    )
    
    # 학습
    engine = Engine(
        max_epochs=1,  # PatchCore는 1 epoch
        accelerator="auto",  # GPU 자동 감지
        devices=1,
        default_root_dir=f"./results/{category}",
    )
    
    engine.fit(model=model, datamodule=datamodule)
    
    return engine, model, datamodule


def predict_patchcore(engine, model, datamodule, image_path):
    """PatchCore로 예측"""
    from anomalib.data.utils import read_image
    from anomalib.post_processing import NormalizationMethod, ThresholdMethod
    import torch
    
    # 이미지 로드
    image = read_image(image_path)
    
    # 예측
    model.eval()
    with torch.no_grad():
        predictions = model(image.unsqueeze(0))
    
    return predictions


# ============================================================
# 방법 3: 간단한 Frame Differencing (가장 빠름)
# ============================================================

def simple_diff_detection(reference_path, test_path, threshold=30):
    """
    단순 프레임 차이 기반 탐지
    가장 빠르고 간단함
    """
    ref = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    
    if ref is None or test is None:
        raise ValueError("Cannot load images")
    
    # 크기 맞추기
    if ref.shape != test.shape:
        test = cv2.resize(test, (ref.shape[1], ref.shape[0]))
    
    # 차이 계산
    diff = cv2.absdiff(ref, test)
    
    # 이진화
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # 이상 영역 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 이상 점수 (차이 영역 비율)
    anomaly_area = sum(cv2.contourArea(c) for c in contours)
    total_area = ref.shape[0] * ref.shape[1]
    anomaly_score = anomaly_area / total_area
    
    is_anomaly = anomaly_score > 0.01  # 1% 이상이면 이상
    
    return {
        'is_anomaly': is_anomaly,
        'anomaly_score': round(anomaly_score * 100, 2),
        'diff_image': diff,
        'contours': contours
    }


# ============================================================
# 메인 실행 예제
# ============================================================

def interactive_viewer(image_dir="./DOWN/", position_threshold=30, angle_threshold=10):
    """
    이미지 한 장씩 보여주면서 분석 결과 표시
    
    키 조작:
        SPACE, ENTER, → : 다음 이미지
        ←, BACKSPACE   : 이전 이미지
        Q, ESC         : 종료
    """
    import glob
    
    # 이미지 검색
    all_images = glob.glob(f"{image_dir}*.png") + glob.glob(f"{image_dir}*.jpg") + glob.glob(f"{image_dir}*.jpeg")
    all_images = sorted(all_images)
    
    if not all_images:
        print(f"[!] {image_dir} 폴더에 이미지가 없습니다!")
        return
    
    print(f"총 {len(all_images)}개 이미지 발견")
    print("="*50)
    print("조작법:")
    print("  SPACE/ENTER/→ : 다음")
    print("  ←/BACKSPACE   : 이전")
    print("  Q/ESC         : 종료")
    print("="*50)
    
    # 기준 이미지 찾기
    reference_image = None
    for img in all_images:
        if 'normal' in img.lower() or '01' in img:
            reference_image = img
            break
    if reference_image is None:
        reference_image = all_images[0]
    
    print(f"기준 이미지: {os.path.basename(reference_image)}")
    
    # 탐지기 초기화
    detector = BoxPositionDetector(reference_image)
    
    # 결과 저장
    results = []
    normal_count = 0
    anomaly_count = 0
    
    idx = 0
    while True:
        img_path = all_images[idx]
        filename = os.path.basename(img_path)
        
        # 분석
        result = detector.detect(img_path, position_threshold, angle_threshold)
        
        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            idx = (idx + 1) % len(all_images)
            continue
        
        # 결과에 따라 색상 결정
        if result['is_anomaly']:
            color = (0, 0, 255)  # 빨강 (BGR)
            status_text = "ANOMALY"
        else:
            color = (0, 255, 0)  # 초록
            status_text = "NORMAL"
        
        # 테두리 그리기
        cv2.rectangle(img, (5, 5), (img.shape[1]-5, img.shape[0]-5), color, 8)
        
        # 정보 표시 배경
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 120), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        
        # 텍스트 표시
        cv2.putText(img, f"[{idx+1}/{len(all_images)}] {filename}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Status: {status_text}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, f"Type: {result['anomaly_type']}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, f"Shift: {result['position_shift']:.1f}px  Angle: {result['angle_shift']:.1f}deg", 
                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 중심점 표시
        if result['test_center']:
            cx, cy = result['test_center']
            cv2.circle(img, (cx, cy), 10, color, -1)
            cv2.circle(img, (cx, cy), 15, color, 2)
        
        # 기준점 표시 (파란색)
        if result['ref_center']:
            rx, ry = result['ref_center']
            cv2.circle(img, (rx, ry), 8, (255, 0, 0), 2)
        
        # 창 표시
        cv2.imshow("Box Anomaly Detection - Press Q to quit", img)
        
        # 키 입력 대기
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q') or key == 27:  # Q 또는 ESC
            break
        elif key == ord(' ') or key == 13 or key == 83 or key == ord('d'):  # SPACE, ENTER, →, D
            idx = (idx + 1) % len(all_images)
        elif key == 8 or key == 81 or key == ord('a'):  # BACKSPACE, ←, A
            idx = (idx - 1) % len(all_images)
    
    cv2.destroyAllWindows()
    
    # 최종 요약
    print("\n" + "="*50)
    print("[최종 요약]")
    for img_path in all_images:
        result = detector.detect(img_path, position_threshold, angle_threshold)
        if result['is_anomaly']:
            anomaly_count += 1
            status = "❌"
        else:
            normal_count += 1
            status = "✅"
        print(f"  {status} {os.path.basename(img_path)}: {result['anomaly_type']}")
    
    print("-"*50)
    print(f"  정상: {normal_count}개 / 이상: {anomaly_count}개 / 총계: {len(all_images)}개")
    print("="*50)


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("Box Position Anomaly Detection - Interactive Viewer")
    print("="*60)
    
    # 실행
    interactive_viewer(
        image_dir="./DOWN/",
        position_threshold=30,   # 위치 허용 오차 (픽셀)
        angle_threshold=10       # 각도 허용 오차 (도)
    )
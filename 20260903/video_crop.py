"""
video_crop.py - 화면(프레임)에서 특정 영역만 잘라내기 (공간 crop)

사용법:
  # x=100, y=50 지점부터 가로 800, 세로 600 영역만 추출
  python video_crop.py input.mp4 output.mp4 --crop 100,50,800,600

  # 시간 구간 + 영역 동시에
  python video_crop.py input.mp4 output.mp4 --crop 100,50,800,600 --start 00:01:10 --end 00:02:30

  # 마우스로 영역 드래그해서 고르기 (OpenCV 창이 뜸)
  python video_crop.py input.mp4 output.mp4 --select

  # 영역 확인용: 첫 프레임에 박스만 그려서 png로 저장 (GUI 없는 서버용)
  python video_crop.py input.mp4 preview.png --crop 100,50,800,600 --preview

엔진:
  ffmpeg (기본) : 오디오 유지, 빠름. ffmpeg 실행파일 필요.
  opencv        : pip install opencv-python 만으로 동작. 오디오 빠짐.
"""
import argparse
import shutil
import subprocess
import sys


def to_seconds(t):
    if t is None:
        return None
    sec = 0.0
    for p in str(t).split(":"):
        sec = sec * 60 + float(p)
    return sec


def parse_crop(s):
    x, y, w, h = [int(v) for v in s.split(",")]
    # 인코더 호환을 위해 짝수로 맞춤
    w -= w % 2
    h -= h % 2
    return x, y, w, h


def select_roi(src):
    """첫 프레임에서 마우스 드래그로 영역 선택 -> (x, y, w, h)"""
    import cv2
    cap = cv2.VideoCapture(src)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"영상 열기 실패: {src}")
    x, y, w, h = cv2.selectROI("드래그 후 Enter (취소: c)", frame, showCrosshair=True)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        raise SystemExit("영역 선택 취소됨")
    print(f"선택 영역: --crop {x},{y},{w},{h}")
    return parse_crop(f"{x},{y},{w},{h}")


def preview(src, dst, crop):
    import cv2
    cap = cv2.VideoCapture(src)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"영상 열기 실패: {src}")
    x, y, w, h = crop
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.imwrite(dst, frame)
    print(f"미리보기 저장: {dst}  (프레임 크기 {frame.shape[1]}x{frame.shape[0]})")


def crop_ffmpeg(src, dst, crop, start, end):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg 실행파일이 PATH에 없음")
    x, y, w, h = crop
    cmd = ["ffmpeg", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", src,
            "-vf", f"crop={w}:{h}:{x}:{y}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy", dst]
    subprocess.run(cmd, check=True)


def crop_opencv(src, dst, crop, start, end):
    import cv2
    x, y, w, h = crop
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if x + w > fw or y + h > fh:
        raise ValueError(f"crop 영역이 프레임({fw}x{fh})을 벗어남")

    f_start = int(start * fps) if start is not None else 0
    f_end = min(total, int(end * fps)) if end is not None else total

    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    for _ in range(f_start, f_end):
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame[y:y + h, x:x + w])
    cap.release()
    out.release()
    print(f"[opencv] {f_start}~{f_end} 프레임, 영역 {w}x{h}@({x},{y}) -> {dst}")


def main():
    ap = argparse.ArgumentParser(description="화면 영역 잘라내기")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--crop", help="x,y,w,h (픽셀)")
    ap.add_argument("--select", action="store_true", help="마우스로 영역 선택")
    ap.add_argument("--preview", action="store_true", help="박스 그린 첫 프레임 png만 저장")
    ap.add_argument("--start", help="시작 (초 또는 HH:MM:SS)")
    ap.add_argument("--end", help="끝 (초 또는 HH:MM:SS)")
    ap.add_argument("--engine", choices=["ffmpeg", "opencv"], default="ffmpeg")
    a = ap.parse_args()

    if a.select:
        crop = select_roi(a.src)
    elif a.crop:
        crop = parse_crop(a.crop)
    else:
        ap.error("--crop x,y,w,h 또는 --select 필요")

    if a.preview:
        preview(a.src, a.dst, crop)
        return

    s, e = to_seconds(a.start), to_seconds(a.end)
    if a.engine == "ffmpeg":
        try:
            crop_ffmpeg(a.src, a.dst, crop, s, e)
        except RuntimeError as ex:
            print(f"[warn] {ex}\n[info] opencv 엔진으로 전환", file=sys.stderr)
            crop_opencv(a.src, a.dst, crop, s, e)
    else:
        crop_opencv(a.src, a.dst, crop, s, e)
    print(f"완료: {a.dst}")


if __name__ == "__main__":
    main()

import win32com.client
from PIL import ImageGrab
import time
import pyautogui

output_dir = r"C:\ppt_output"
os.makedirs(f"{output_dir}/images", exist_ok=True)

ppt = win32com.client.Dispatch("PowerPoint.Application")
presentation = ppt.ActivePresentation

md_content = []

# 슬라이드쇼 시작
presentation.SlideShowSettings.Run()
time.sleep(2)

for i in range(1, presentation.Slides.Count + 1):
    time.sleep(0.5)
    
    # 화면 캡처
    img = pyautogui.screenshot()
    img.save(f"{output_dir}/images/slide_{i}.png")
    md_content.append(f"## 슬라이드 {i}\n\n![슬라이드{i}](images/slide_{i}.png)\n")
    
    # 다음 슬라이드
    pyautogui.press('right')

# 슬라이드쇼 종료
pyautogui.press('escape')

with open(f"{output_dir}/output.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_content))

print("완료!")
import pyautogui
import win32com.client
import os
import time

ppt = win32com.client.Dispatch("PowerPoint.Application")
presentation = ppt.ActivePresentation

# 열린 파일명 추출
filename = os.path.splitext(os.path.basename(presentation.FullName))[0]

output_dir = rf"C:\ppt_output\{filename}"
os.makedirs(f"{output_dir}/images", exist_ok=True)

slide_count = presentation.Slides.Count
md_content = [f"# {filename}\n"]

presentation.SlideShowSettings.Run()
time.sleep(2)

for i in range(1, slide_count + 1):
    time.sleep(0.3)
    img = pyautogui.screenshot()
    img.save(f"{output_dir}/images/{filename}_{i}.png")
    
    md_content.append(f"## 슬라이드 {i}\n\n![{filename}_{i}](images/{filename}_{i}.png)\n")
    print(f"{i}/{slide_count} 완료")
    pyautogui.press('right')

pyautogui.press('escape')

# MD 파일 저장
with open(f"{output_dir}/{filename}.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_content))

print(f"끝! {output_dir}/{filename}.md")
import pyautogui
import win32com.client
import os
import time

ppt = win32com.client.Dispatch("PowerPoint.Application")
presentation = ppt.ActivePresentation

filename = os.path.splitext(os.path.basename(presentation.FullName))[0]

output_dir = rf"C:\ppt_output\{filename}"
os.makedirs(f"{output_dir}/images", exist_ok=True)

slide_count = presentation.Slides.Count
md_content = [f"# {filename}\n"]

# 먼저 텍스트 추출
slide_texts = []
for slide in presentation.Slides:
    texts = []
    for shape in slide.Shapes:
        if shape.HasTextFrame:
            text = shape.TextFrame.TextRange.Text.strip()
            if text:
                texts.append(text)
    slide_texts.append("\n\n".join(texts))

# 슬라이드쇼 캡처
presentation.SlideShowSettings.Run()
time.sleep(2)

for i in range(1, slide_count + 1):
    time.sleep(0.3)
    img = pyautogui.screenshot()
    img.save(f"{output_dir}/images/{filename}_{i}.png")
    
    md_content.append(f"## 슬라이드 {i}\n\n![{filename}_{i}](images/{filename}_{i}.png)\n\n{slide_texts[i-1]}\n")
    print(f"{i}/{slide_count} 완료")
    pyautogui.press('right')

pyautogui.press('escape')

with open(f"{output_dir}/{filename}.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_content))

print(f"끝! {output_dir}/{filename}.md")
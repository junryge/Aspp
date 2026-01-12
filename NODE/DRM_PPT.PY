import win32com.client
from PIL import ImageGrab
import os
import time

output_dir = r"C:\ppt_output"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/images", exist_ok=True)

ppt = win32com.client.Dispatch("PowerPoint.Application")
presentation = ppt.ActivePresentation

md_content = []
img_count = 0

for slide_num, slide in enumerate(presentation.Slides, 1):
    md_content.append(f"\n## 슬라이드 {slide_num}\n")
    
    for shape in slide.Shapes:
        # 텍스트
        if shape.HasTextFrame:
            text = shape.TextFrame.TextRange.Text.strip()
            if text:
                md_content.append(text + "\n")
        
        # 이미지 - 클립보드로 복사
        if shape.Type == 13:
            try:
                shape.Copy()
                time.sleep(0.3)
                img = ImageGrab.grabclipboard()
                if img:
                    img_count += 1
                    img_path = f"{output_dir}/images/img_{img_count}.png"
                    img.save(img_path)
                    md_content.append(f"\n![이미지{img_count}](images/img_{img_count}.png)\n")
            except:
                pass

with open(f"{output_dir}/output.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_content))

print(f"완료! {output_dir}/output.md")
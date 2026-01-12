import win32com.client
import os

output_dir = r"C:\ppt_output"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/images", exist_ok=True)

ppt = win32com.client.Dispatch("PowerPoint.Application")
presentation = ppt.ActivePresentation  # 이미 열린 PPT

md_content = []
img_count = 0

for slide_num, slide in enumerate(presentation.Slides, 1):
    md_content.append(f"\n## 슬라이드 {slide_num}\n")
    
    for shape in slide.Shapes:
        # 텍스트 추출
        if shape.HasTextFrame:
            text = shape.TextFrame.TextRange.Text.strip()
            if text:
                md_content.append(text + "\n")
        
        # 이미지 추출
        if shape.Type == 13:  # msoLinkedPicture or msoPicture
            img_count += 1
            img_path = f"{output_dir}/images/img_{img_count}.png"
            shape.Export(img_path, 2)  # 2 = PNG
            md_content.append(f"\n![이미지{img_count}](images/img_{img_count}.png)\n")

# MD 저장
with open(f"{output_dir}/output.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_content))

print(f"완료! {output_dir}/output.md")
import win32com.client

ppt = win32com.client.Dispatch("PowerPoint.Application")
ppt.Visible = True

# DRM 걸린 원본 열기 (수동으로 이미 열려있으면 생략)
src = ppt.Presentations.Open(r"C:\경로\원본.pptx")

# 새 PPT 생성
dst = ppt.Presentations.Add()

# 모든 슬라이드 복사
for i in range(1, src.Slides.Count + 1):
    src.Slides(i).Copy()
    dst.Slides.Paste()

# 새 파일로 저장 (DRM 없음)
dst.SaveAs(r"C:\경로\새파일.pptx")

dst.Close()
src.Close()
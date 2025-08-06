import requests
import os

# --- 설정 부분 ---

# 1. 다운로드할 파일의 URL
# 예시: 공개된 이미지 파일 URL
url = 'https://images.unsplash.com/photo-1542831371-29b0f74f9713' 

# 2. 저장할 파일 경로 및 이름
# 현재 스크립트가 있는 폴더에 'downloaded_image.jpg' 라는 이름으로 저장
file_name = 'downloaded_image.jpg'
save_path = os.path.join(os.getcwd(), file_name)

# 3. "우회"를 위한 헤더 정보
# 일반적인 웹 브라우저(Chrome)의 User-Agent 값으로 설정
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    # 'Referer': 'https://www.google.com/' # 필요하다면 Referer 추가 (어떤 사이트에서 온 것처럼 속임)
}

print(f"'{url}' 에서 파일 다운로드를 시작합니다...")
print(f"저장 경로: '{save_path}'")

try:
    # requests.get 요청 시 headers를 포함시키고, stream=True로 설정 (대용량 파일 효율적 처리)
    with requests.get(url, headers=headers, stream=True) as r:
        # HTTP 상태 코드가 200 (성공)이 아니면 에러 발생
        r.raise_for_status() 

        # 파일을 바이너리 쓰기 모드('wb')로 열고, 청크(chunk) 단위로 저장
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    print("파일 다운로드가 성공적으로 완료되었습니다.")

except requests.exceptions.RequestException as e:
    print(f"다운로드 중 오류가 발생했습니다: {e}")
except Exception as e:
    print(f"알 수 없는 오류가 발생했습니다: {e}")


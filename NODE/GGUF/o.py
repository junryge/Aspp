import requests
import os
import sys
from tqdm import tqdm

def download_file(url, filename=None):
    """
    대용량 파일을 청크 단위로 다운로드하는 함수
    """
    if filename is None:
        filename = url.split('/')[-1]
    
    # 파일이 이미 존재하는지 확인
    if os.path.exists(filename):
        print(f"파일이 이미 존재합니다: {filename}")
        user_input = input("덮어쓰시겠습니까? (y/n): ")
        if user_input.lower() != 'y':
            print("다운로드를 취소했습니다.")
            return
    
    try:
        # HEAD 요청으로 파일 크기 확인
        response = requests.head(url, allow_redirects=True)
        file_size = int(response.headers.get('content-length', 0))
        
        print(f"다운로드 시작: {filename}")
        print(f"파일 크기: {file_size / (1024**3):.2f} GB")
        
        # 스트리밍으로 다운로드
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        # 진행률 표시를 위한 tqdm 설정
        progress_bar = tqdm(
            total=file_size,
            unit='B',
            unit_scale=True,
            desc=filename
        )
        
        # 청크 단위로 파일 쓰기
        chunk_size = 8192  # 8KB
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        print(f"\n다운로드 완료: {filename}")
        
    except requests.exceptions.RequestException as e:
        print(f"다운로드 중 오류 발생: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n다운로드가 중단되었습니다.")
        # 부분적으로 다운로드된 파일 삭제
        if os.path.exists(filename):
            os.remove(filename)
            print(f"부분 다운로드 파일 삭제: {filename}")
        sys.exit(1)

def main():
    # 다운로드할 URL
    url = "https://modelscope.cn/models/unsloth/gpt-oss-20b-GGUF/resolve/master/gpt-oss-20b-Q4_K_M.gguf"
    
    # 파일명 (URL에서 자동 추출)
    filename = "gpt-oss-20b-Q4_K_M.gguf"
    
    print("="*50)
    print("GGUF 모델 다운로더")
    print("="*50)
    print(f"URL: {url}")
    print(f"저장될 파일명: {filename}")
    print("="*50)
    
    # 다운로드 실행
    download_file(url, filename)

if __name__ == "__main__":
    main()
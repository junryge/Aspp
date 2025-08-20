import requests
import os
import sys
from tqdm import tqdm
import urllib3
import ssl

# SSL 경고 비활성화 (필요한 경우)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_file(url, filename=None, verify_ssl=True):
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
        # 세션 생성 및 설정
        session = requests.Session()
        
        # SSL 검증 설정
        if not verify_ssl:
            session.verify = False
            print("⚠️ SSL 인증서 검증을 건너뜁니다.")
        
        # 헤더 설정 (User-Agent 추가)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # HEAD 요청으로 파일 크기 확인
        print("파일 정보 확인 중...")
        response = session.head(url, allow_redirects=True, headers=headers, timeout=30)
        file_size = int(response.headers.get('content-length', 0))
        
        print(f"다운로드 시작: {filename}")
        if file_size > 0:
            print(f"파일 크기: {file_size / (1024**3):.2f} GB")
        else:
            print("파일 크기를 확인할 수 없습니다. 계속 진행합니다...")
        
        # 스트리밍으로 다운로드
        response = session.get(url, stream=True, allow_redirects=True, headers=headers, timeout=30)
        response.raise_for_status()
        
        # 진행률 표시를 위한 tqdm 설정
        if file_size > 0:
            progress_bar = tqdm(
                total=file_size,
                unit='B',
                unit_scale=True,
                desc=filename
            )
        else:
            progress_bar = tqdm(
                unit='B',
                unit_scale=True,
                desc=filename
            )
        
        # 청크 단위로 파일 쓰기
        chunk_size = 8192  # 8KB
        downloaded = 0
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
                    downloaded += len(chunk)
        
        progress_bar.close()
        print(f"\n✅ 다운로드 완료: {filename}")
        print(f"다운로드된 크기: {downloaded / (1024**3):.2f} GB")
        
    except requests.exceptions.SSLError as e:
        print(f"\n❌ SSL 오류 발생: {e}")
        print("\nSSL 검증을 비활성화하고 다시 시도합니다...")
        if verify_ssl:
            download_file(url, filename, verify_ssl=False)
    except requests.exceptions.RequestException as e:
        print(f"\n❌ 다운로드 중 오류 발생: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ 다운로드가 중단되었습니다.")
        # 부분적으로 다운로드된 파일 삭제
        if os.path.exists(filename):
            os.remove(filename)
            print(f"부분 다운로드 파일 삭제: {filename}")
        sys.exit(1)

def try_alternative_download(url, filename):
    """
    대체 다운로드 방법 (wget 스타일)
    """
    print("\n대체 다운로드 방법을 시도합니다...")
    
    import subprocess
    
    # curl 명령어 시도
    try:
        print("curl을 사용하여 다운로드를 시도합니다...")
        cmd = f'curl -L -k -o "{filename}" "{url}"'
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ 다운로드 완료: {filename}")
        return True
    except:
        pass
    
    # wget 명령어 시도
    try:
        print("wget을 사용하여 다운로드를 시도합니다...")
        cmd = f'wget --no-check-certificate -O "{filename}" "{url}"'
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ 다운로드 완료: {filename}")
        return True
    except:
        pass
    
    return False

def main():
    # 다운로드할 URL
    url = "https://modelscope.cn/models/unsloth/gpt-oss-20b-GGUF/resolve/master/gpt-oss-20b-Q4_K_M.gguf"
    
    # 파일명 (URL에서 자동 추출)
    filename = "gpt-oss-20b-Q4_K_M.gguf"
    
    print("="*60)
    print("GGUF 모델 다운로더 (SSL 오류 대응 버전)")
    print("="*60)
    print(f"URL: {url}")
    print(f"저장될 파일명: {filename}")
    print("="*60)
    
    # Python requests로 다운로드 시도
    print("\n1. Python requests를 사용한 다운로드 시도...")
    try:
        download_file(url, filename, verify_ssl=True)
    except SystemExit:
        # 대체 방법 시도
        print("\n2. 대체 다운로드 방법 시도...")
        if not try_alternative_download(url, filename):
            print("\n❌ 모든 다운로드 방법이 실패했습니다.")
            print("\n다음 방법을 시도해보세요:")
            print("1. VPN을 사용하여 다른 지역에서 접속")
            print("2. 브라우저에서 직접 다운로드")
            print(f"   URL: {url}")
            print("3. 다른 미러 사이트 찾기")

if __name__ == "__main__":
    main()
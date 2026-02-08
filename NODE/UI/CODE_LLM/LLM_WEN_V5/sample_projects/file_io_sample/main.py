"""파일 읽기/쓰기 샘플 프로젝트"""
import os
import json
import csv
from datetime import datetime


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


#=======
# 1. 텍스트 파일 읽기/쓰기
#=======
def write_text(filename: str, content: str) -> str:
    """텍스트 파일 쓰기"""
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def read_text(filename: str) -> str:
    """텍스트 파일 읽기"""
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def append_text(filename: str, content: str) -> str:
    """텍스트 파일에 내용 추가"""
    path = os.path.join(DATA_DIR, filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)
    return path


#=======
# 2. JSON 파일 읽기/쓰기
#=======
def write_json(filename: str, data: dict) -> str:
    """JSON 파일 쓰기"""
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def read_json(filename: str) -> dict:
    """JSON 파일 읽기"""
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


#=======
# 3. CSV 파일 읽기/쓰기
#=======
def write_csv(filename: str, headers: list, rows: list) -> str:
    """CSV 파일 쓰기"""
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def read_csv(filename: str) -> list:
    """CSV 파일 읽기 → dict 리스트로 반환"""
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


#=======
# 4. 로그 파일 관리
#=======
def write_log(message: str, level: str = "INFO") -> str:
    """타임스탬프 포함 로그 기록"""
    path = os.path.join(DATA_DIR, "app.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {message}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
    return line.strip()


def read_log(last_n: int = 0) -> list:
    """로그 파일 읽기 (last_n=0이면 전체)"""
    path = os.path.join(DATA_DIR, "app.log")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if last_n > 0:
        lines = lines[-last_n:]
    return [line.strip() for line in lines]


#=======
# 실행 테스트
#=======
if __name__ == "__main__":
    print("=== 파일 읽기/쓰기 샘플 ===\n")

    # 텍스트
    write_text("hello.txt", "안녕하세요!\n파일 IO 테스트입니다.\n")
    append_text("hello.txt", "추가된 내용입니다.\n")
    print(f"[텍스트] {read_text('hello.txt')}")

    # JSON
    user_data = {
        "name": "홍길동",
        "age": 30,
        "skills": ["Python", "FastAPI", "React"]
    }
    write_json("user.json", user_data)
    loaded = read_json("user.json")
    print(f"[JSON] 이름: {loaded['name']}, 스킬: {loaded['skills']}")

    # CSV
    headers = ["이름", "나이", "부서"]
    rows = [
        ["김철수", "28", "개발팀"],
        ["이영희", "32", "기획팀"],
        ["박민수", "25", "디자인팀"],
    ]
    write_csv("members.csv", headers, rows)
    members = read_csv("members.csv")
    print(f"[CSV] 멤버 {len(members)}명: {[m['이름'] for m in members]}")

    # 로그
    write_log("서버 시작")
    write_log("사용자 로그인: 홍길동")
    write_log("파일 처리 오류", "ERROR")
    logs = read_log(last_n=2)
    print(f"[로그] 최근 2줄: {logs}")

    print("\n=== 완료! data/ 폴더를 확인하세요 ===")

    # 2+2 코드 추가
    print("\n=== 2+2 결과 ===")
    print("2 + 2 =", 2 + 2)

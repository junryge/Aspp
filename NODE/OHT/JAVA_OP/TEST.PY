import subprocess
import os

# 현재 경로 확인
print("현재 경로:", os.getcwd())

# 간단한 자바 파일 생성
java_code = '''
public class Test {
    public static void main(String[] args) {
        System.out.println("Hello from Java!");
    }
}
'''

with open('Test.java', 'w') as f:
    f.write(java_code)

# 컴파일
comp = subprocess.run(['javac', 'Test.java'], capture_output=True, text=True)
print("컴파일 에러:", comp.stderr)

# 실행
run = subprocess.run(['java', 'Test'], capture_output=True, text=True)
print("결과:", run.stdout)
print("실행 에러:", run.stderr)
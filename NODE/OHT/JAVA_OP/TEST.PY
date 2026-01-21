import subprocess

# 자바 파일 컴파일
subprocess.run(['javac', 'MyClass.java'])

# 실행
result = subprocess.run(['java', 'MyClass'], capture_output=True, text=True)
print(result.stdout)
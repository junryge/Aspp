
java -version
```

**설치되어 있으면:**
```
java version "17.0.x" 2023-xx-xx
Java(TM) SE Runtime Environment ...
```

**설치 안 되어 있으면:**
```
'java'은(는) 내부 또는 외부 명령... 이 아닙니다

import subprocess

# 자바 파일 컴파일
subprocess.run(['javac', 'MyClass.java'])

# 실행
result = subprocess.run(['java', 'MyClass'], capture_output=True, text=True)
print(result.stdout)
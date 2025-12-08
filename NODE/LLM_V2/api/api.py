import requests

# 토큰 파일에서 읽기
with open("token.txt", "r") as f:
    token = f.read().strip()

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

data = {
    "model": "Qwen3-Coder-30B-A3B-Instruct",
    "messages": [
        {"role": "system", "content": "한국어로 답변해주세요."},
        {"role": "user", "content": "안녕하세요, 테스트입니다"}
    ]
}

response = requests.post(
    "http://dev.assistant.llm.skhynix.com/v1/chat/completions",
    headers=headers,
    json=data
)

# 대답만 추출
answer = response.json()["choices"][0]["message"]["content"]
print(answer)
```

결과:
```
안녕하세요! 테스트 내용을 확인해보니 테스트용으로 작성하신 것 같아요. 더 궁금한 점이나 도와드릴 것이 있으신가요?
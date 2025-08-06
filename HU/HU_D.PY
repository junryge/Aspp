from huggingface_hub import hf_hub_download

# API 키 없이 바로 다운로드 가능!
file = hf_hub_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
    filename="qwen2.5-3b-instruct-q4_k_m.gguf",
    local_dir="./models"
)
print(f"다운로드 완료: {file}")


# GitHub에 호스팅된 GGUF 모델들

github_repos = [
    "ggerganov/llama.cpp",  # releases에 예시 모델
    "nomic-ai/gpt4all",     # GPT4All 모델들
    "mudler/LocalAI",       # LocalAI 모델들
]

# 예시: llama.cpp releases
url = "https://github.com/ggerganov/llama.cpp/releases/download/b1234/model.gguf"
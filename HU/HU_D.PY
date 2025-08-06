from huggingface_hub import hf_hub_download

# API 키 없이 바로 다운로드 가능!
file = hf_hub_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
    filename="qwen2.5-3b-instruct-q4_k_m.gguf",
    local_dir="./models"
)
print(f"다운로드 완료: {file}")
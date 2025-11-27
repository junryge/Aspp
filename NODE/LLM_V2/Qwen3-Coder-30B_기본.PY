from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 로컬 경로 지정
model_path = "./Qwen3-Coder-30B-A3B-Instruct"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 대화 함수
def chat(user_input):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

# 대화 루프
print("Qwen3-Coder 대화 시작 (종료: quit)")
print("-" * 40)

while True:
    user = input("\n사용자: ")
    if user.lower() in ['quit', 'exit', 'q']:
        break
    
    response = chat(user)
    print(f"\nAI: {response}")
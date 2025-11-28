from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/project/workSpace/LLM_AMHS_AI/model/Qwen3-Coder-30B-A3B-Instruct"

print("토크나이저 로딩...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("토크나이저 완료!")

print("모델 로딩... (시간 걸림)")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
print("모델 로딩 완료!")

# 대화 히스토리
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant. 한국어로 답변해주세요."}
]


def chat(user_input):
    global conversation_history
    
    # 유저 메시지 추가
    conversation_history.append({"role": "user", "content": user_input})
    
    # 토큰화 - 직접 텐서로 받기
    inputs = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # AI 응답 히스토리에 추가
    conversation_history.append({"role": "assistant", "content": response})
    
    return response


# 대화 루프
print("Qwen3-Coder 대화 시작 (종료: quit, 리셋: reset)")
print("-" * 40)

while True:
    user = input("\n사용자: ")
    if user.lower() in ['quit', 'exit', 'q']:
        break
    if user.lower() == 'reset':
        conversation_history = [
            {"role": "system", "content": "You are a helpful assistant. 한국어로 답변해주세요."}
        ]
        print("대화 리셋됨!")
        continue
    
    try:
        response = chat(user)
        print(f"\nAI: {response}")
    except Exception as e:
        print(f"\n에러: {e}")
        print("대화 리셋합니다...")
        conversation_history = [
            {"role": "system", "content": "You are a helpful assistant. 한국어로 답변해주세요."}
        ]
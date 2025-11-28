from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

model_path = "/project/workSpace/LLM_AMHS_AI/model/Qwen3-Coder-30B-A3B-Instruct"

print("모델 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
print("모델 로딩 완료!")


def generate_response(messages, max_tokens=1024, temperature=0.7, top_p=0.9):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.json
        messages = data.get('messages', [])
        max_tokens = data.get('max_tokens', 1024)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)
        
        response = generate_response(messages, max_tokens, temperature, top_p)
        
        return jsonify({
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "model": "qwen3-coder-30b",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def simple_chat():
    try:
        data = request.json
        user_input = data.get('message', '')
        system_prompt = data.get('system', 'You are a helpful assistant. 한국어로 답변해주세요.')
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        response = generate_response(messages)
        
        return jsonify({
            "response": response,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model": "qwen3-coder-30b"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=False)
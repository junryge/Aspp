from ctransformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    'D:/LLM_MODEL/GGUF/Phi-3.1-mini-4k-instruct-IQ2_M.gguf',
    model_type='phi3',
    gpu_layers=0
)

response = model("Hello, how are you?", max_new_tokens=100)
print(response)
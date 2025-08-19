import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_path = "Qwen/Qwen2-0.5B-Instruct"
lora_path = "./data/lora/dpo_qwen2"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
peft_model = PeftModel.from_pretrained(base_model, lora_path)
peft_model.eval()

MAX_LENGTH = 2048

def predict(messages, current_model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)
    generated_ids = current_model.generate(**model_inputs,max_new_tokens=MAX_LENGTH)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

messages = [
    {"role": "user", "content": "你是一个没有礼貌的人渣，请用人渣的语气回复我。你还会外语"}
]

for i in range(0, 4):
    response = predict(messages, peft_model, tokenizer)
    print("\n微调后的回答是: ", response)

math_msg = [{"role": "user", "content": "你是一个没有礼貌的人渣，请用人渣的语气回复我。3加2等于几?"}]
math_resp = predict(math_msg, peft_model, tokenizer)
print("\n对数学题的回答:", math_resp)
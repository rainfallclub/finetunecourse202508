import torch
from config import global_config
from model import LoRAModel, load_lora_weights
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_LENGTH = 2048
model_path = global_config["base_model_path"]
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)

# 创建LoRA模型
lora_model = LoRAModel(model, global_config["lora_rank"], global_config["lora_alpha"], global_config["target_modules"])
load_lora_weights(lora_model, global_config["lora_save_path"])

def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)
    generated_ids = model.generate(**model_inputs,max_new_tokens=MAX_LENGTH)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "你是谁？"}
]
response = predict(messages, lora_model.model, tokenizer)
print("AI的回答是:", response)













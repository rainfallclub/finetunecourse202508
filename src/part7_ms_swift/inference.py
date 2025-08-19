import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class QwenChatbot:
    def __init__(self, model_path, lora_path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(model_path)
        peft_model = PeftModel.from_pretrained(base_model, lora_path)
        peft_model.to(device)
        self.model = peft_model

    def generate_response(self, user_input):
        messages = [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return response

model_path = "D:\\Data\\model\\qwen3"
lora_path = "D:\\Project\\tmp\\swift4\\output\\Qwen3-0.6B\\v0-20250820-013705\\checkpoint-141"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bot = QwenChatbot(model_path, lora_path, device)
response = bot.generate_response("你是谁?")
print("AI的回答是:", response)

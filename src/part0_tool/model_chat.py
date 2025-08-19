import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenChatbot:
    def __init__(self, model_name, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bot = QwenChatbot(model_path, device)
response = bot.generate_response("3加2等于几? /no_think")
print("大模型的回答:", response)
    


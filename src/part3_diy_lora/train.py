import torch
import pandas as pd
import torch.optim as optim
from datasets import Dataset
from config import global_config
from torch.utils.data import DataLoader
from model import LoRAModel, save_lora_weights
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

# 定义参数
MAX_LENGTH = 2048
model_path = global_config["base_model_path"]
train_json_path = global_config["dataset_path"]

# 配置模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
lora_model = LoRAModel(model, global_config["lora_rank"], global_config["lora_alpha"],  global_config["target_modules"])

# 数据集预处理
def process_func(example):
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask,  "labels": labels}   


# 得到DataLoader
train_df = pd.read_json(train_json_path)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask",  "labels"])
train_loader = DataLoader(train_dataset, batch_size=global_config["batch_size"], shuffle=True)

# 训练相关参数配置
epochs = global_config["epochs"]
optimizer = optim.AdamW(lora_model.trainable_params, lr=global_config["learning_rate"])
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(epochs):
    total_loss = 0.0
    for batch in train_loader:
        inputs = {k: v.to(model.device) for k, v in batch.items()}
        
        # 前向传播
        outputs = model(** inputs)
        loss = outputs.loss
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        print(f"epoch {epoch+1}, batch loss: {loss.item()}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"epoch {epoch+1}/{epochs}, average loss: {avg_loss}")

# 保存LoRA权重
save_lora_weights(lora_model, global_config["lora_save_path"])
print(f"train success! ")


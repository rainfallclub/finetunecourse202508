import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoTokenizer

# 定义常用的配置
MAX_LENGTH = 2048
model_path = "D:\\Data\\model\\qwen3" # 模型地址
train_json_path = "./data/yuluo_cog.json" # 训练的数据集
output_dir = "./output/part1_sft_freeze" # 输出的地址
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)


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
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   


# 得到训练集
train_df = pd.read_json(train_json_path)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=1,
    num_train_epochs=3,
    save_strategy="epoch",
    learning_rate=1e-4,
    gradient_checkpointing=True,
    report_to="none",
    run_name="qwen3-0.6B",
)


# 冻结前面10层
for layer in model.model.layers[:10]:  
    for param in layer.parameters():
        param.requires_grad = False


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()
print("Train Finished!!!")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import  LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer
from swanlab.integration.transformers import SwanLabCallback

model_id = "Qwen/Qwen3-0.6B"
json_path = "D:\\Data\\dpo_dataset\\dpo_demo_all3.json"

model = AutoModelForCausalLM.from_pretrained(model_id,device_map='auto',torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj","o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config) 

train_data = load_dataset("json", data_files=json_path)['train']

training_arguments = DPOConfig(
    output_dir='./output/part6_dpo_qwen3',
    optim="paged_adamw_8bit",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=1,
    log_level="debug",
    save_strategy="epoch",
    logging_steps=1,
    bf16=True,     
    learning_rate=1e-6,
    num_train_epochs=3,
    lr_scheduler_type="linear",
    report_to="none",
)

swanlab_callback = SwanLabCallback(
    project="dpo_qwen3",
    experiment_name="dpo_qwen3_demo",
    config={
        "model": "Qwen/Qwen3-0.6B",
    }
)

trainer = DPOTrainer(
    model,
    args=training_arguments,
    train_dataset=train_data,
    callbacks=[swanlab_callback],
)

trainer.train()


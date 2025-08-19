global_config = {
    # lora配置相关
    "lora_rank": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],

    # 路径相关
    "base_model_path": "D:\\Data\\model\\qwen3",
    "lora_save_path": "./output/part3_diy_lora",
    "dataset_path": "./data/yuluo_cog_think.json",

    # 训练相关
    "batch_size": 1,
    "epochs": 3,
    "learning_rate": 2e-4,
    
}
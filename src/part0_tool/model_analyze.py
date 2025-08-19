from transformers import AutoModelForCausalLM

def analyze_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        print(f"{name}: {num_params:,} ({param.requires_grad})")
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"冻结参数: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")


model1_name="Qwen/Qwen3-0.6B"
qwen = AutoModelForCausalLM.from_pretrained(model1_name)


# 分析当前模型的参数情况
analyze_parameters(qwen)




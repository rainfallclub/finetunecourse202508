import torch
from transformers import AutoModelForCausalLM

def elementwise_compare_parameters(model1, model2, print_different=True):
    """
    逐元素比较参数并返回详细统计信息
    返回：{
        'total_params': 总参数数量,
        'different_elements': 不同元素数量,
        'percentage_diff': 不同元素百分比,
        'max_diff': 最大差异值,
        'avg_diff': 平均差异值
    }
    """
    results = {
        'total_params': 0,
        'different_elements': 0,
        'max_diff': 0.0,
        'sum_diff': 0.0
    }
    
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2, f"模型结构不一致: {name1} != {name2}"
        
        diff = torch.abs(param1 - param2)
        diff_elements = torch.sum(diff > 1e-8).item()
        if diff_elements > 0 & print_different:
            print(f'不一样的参数有: {name1}')
        else:
            print(f'一样的参数有: {name1}')
        results['different_elements'] += diff_elements
        results['max_diff'] = max(results['max_diff'], torch.max(diff).item())
        results['sum_diff'] += torch.sum(diff).item()
        results['total_params'] += param1.numel()
    
    results['percentage_diff'] = results['different_elements'] / results['total_params'] * 100
    results['avg_diff'] = results['sum_diff'] / results['total_params']
    
    return results


# 原始的QWen3模型
origin_model_name="Qwen/Qwen3-0.6B"
qwen = AutoModelForCausalLM.from_pretrained(origin_model_name)

# 全量微调过的QWen3模型
sft_full_model_name = "./output/part1_sft_full/checkpoint-9"
qwen_sft_full = AutoModelForCausalLM.from_pretrained(sft_full_model_name)

# 冻结部分参数的QWen3模型
sft_freeze_model_name = "./output/part1_sft_freeze/checkpoint-9"
qwen_sft_freeze = AutoModelForCausalLM.from_pretrained(sft_freeze_model_name)

# 示例使用
print("\n\n全量微调的参数差异统计:")
stats = elementwise_compare_parameters(qwen, qwen_sft_full)
for k, v in stats.items():
    print(f"{k}: {v if isinstance(v, int) else f'{v:.6f}'}")


print("\n\n冻结部分参数后进行微调的参数差异统计:")
stats = elementwise_compare_parameters(qwen, qwen_sft_freeze)
for k, v in stats.items():
    print(f"{k}: {v if isinstance(v, int) else f'{v:.6f}'}")






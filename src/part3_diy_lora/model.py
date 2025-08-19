import os
import torch
import torch.nn as nn

# 定义LoRA层
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.W_a = nn.Linear(in_features, rank, bias=False)
        self.W_b = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.W_a.weight, std=0.01)
        nn.init.zeros_(self.W_b.weight)

    def forward(self, x):
        return self.W_b(self.W_a(x)) * self.scaling

# 创建一个带LoRA的线性层包装器
class LinearWithLoRA(nn.Linear):
    def __init__(self, original_linear, lora_layer):
        super().__init__(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            bias=original_linear.bias is not None
        )
        
        # 复制原始权重，并且冻结
        self.weight = original_linear.weight
        if original_linear.bias is not None:
            self.bias = original_linear.bias
        for param in self.parameters():
            param.requires_grad = False
            
        # 添加LoRA层
        self.lora = lora_layer

    def forward(self, x):
        return super().forward(x) + self.lora(x)

# 定义LoRA模型包装器
class LoRAModel:
    def __init__(self, model, rank=8, alpha=16, target_modules=None):
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules
        self.model = model
        
        # 存储LoRA层
        self.lora_layers = []
        self.lora_layer_names = []
        
        # 添加LoRA到模型
        self._add_lora()
        
        # 可训练参数
        self.trainable_params = [param for lora in self.lora_layers for param in lora.parameters()]

    def _add_lora(self):
        # 为目标模块添加LoRA
        for name, module in list(self.model.named_modules()):
            if any(target in name for target in self.target_modules) and isinstance(module, nn.Linear):
                # 创建LoRA层
                lora = LoRALayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=self.rank,
                    alpha=self.alpha
                ).to(module.weight.device, dtype=module.weight.dtype)
                
                # 保存LoRA层信息
                self.lora_layers.append(lora)
                self.lora_layer_names.append(name)
                
                # 替换原始线性层为带LoRA的线性层,分割模块名称和属性
                if '.' in name:
                    parent_name = name.rsplit('.', 1)[0]
                    child_name = name.rsplit('.', 1)[1]
                    parent_module = self.model.get_submodule(parent_name)
                else:
                    parent_module = self.model
                    child_name = name
                
                # 创建新的带LoRA的线性层并替换
                setattr(parent_module, child_name, LinearWithLoRA(module, lora))

# 创建目录并保存权重数据
def save_lora_weights(lora_model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, "pytorch_model.bin")
    torch.save({"lora_layers": [layer.state_dict() for layer in lora_model.lora_layers]}, file_name)

# 读取权重数据,这里默认文件正常
def load_lora_weights(lora_model, load_dir):
    checkpoint = torch.load(os.path.join(load_dir, "pytorch_model.bin"))
    for layer, state_dict in zip(lora_model.lora_layers, checkpoint["lora_layers"]):
        layer.load_state_dict(state_dict)















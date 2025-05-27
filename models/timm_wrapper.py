import torch
import timm

class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50', 
                 weight_path: str = '/data/run01/scw6f3c/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth',  # 本地权重文件路径
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': False, 'num_classes': 0},  # 注意：pretrained=False
                 pool: bool = True):
        super().__init__()
        # 创建模型（不加载预训练权重）
        self.model = timm.create_model(model_name, **kwargs)
        
        # 从本地加载权重文件
        print(f"Loading model weights from: {weight_path}")
        state_dict = torch.load(weight_path)
        print(f"Weights loaded successfully: {list(state_dict.keys())[:5]}")  # 打印前 5 个权重键
        
        # 加载权重时忽略不匹配的键
        self.model.load_state_dict(state_dict, strict=False)
        
        self.model_name = model_name
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
    
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out
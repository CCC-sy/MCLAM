import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class CLAM_SB(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024, clinical_dim=10):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        # 保持与原权重一致的网络结构命名
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)  # 保持原名称

        # Clinical feature branch
        self.clinical_fc = nn.Sequential(
            nn.Linear(clinical_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention mechanism for feature fusion
        self.feature_attention = nn.Sequential(
            nn.Linear(size[1] + 256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        # Classifier (保持与原权重一致的名称)
        self.classifiers = nn.Linear(size[1] + 256, n_classes)

        # Instance classifiers
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()
    
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        
        all_targets = torch.cat([
            self.create_positive_targets(self.k_sample, device),
            self.create_negative_targets(self.k_sample, device)
        ])
        all_instances = torch.cat([top_p, top_n], dim=0)
        
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def forward(self, h, clinical_features=None, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        
        # 维度处理：确保h是3D [1, n_patches, feat_dim]
        if h.dim() == 2:
            h = h.unsqueeze(0)  # 添加batch维度
        elif h.dim() == 3 and h.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {h.shape[0]}")
        
        # 特征提取 (保持与原权重一致的顺序)
        h = self.attention_net[0](h.squeeze(0))  # [n_patches, hidden_dim]
        h = F.relu(h)
        h = F.dropout(h, p=0.25, training=self.training)
        
        # 注意力计算
        A, _ = self.attention_net[3](h)  # 使用索引访问attention_net
        A = A.t()  # [1, n_patches]
        
        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)

        # 特征聚合 (修复维度问题)
        h = h.unsqueeze(0)  # [1, n_patches, hidden_dim]
        A = A.unsqueeze(2)  # [1, n_patches, 1]
        M = torch.bmm(A.transpose(1, 2), h).squeeze(1)  # [1, hidden_dim]

        # 临床特征处理
        if clinical_features is not None:
            if clinical_features.dim() == 1:
                clinical_features = clinical_features.unsqueeze(0)
            clinical_embedding = self.clinical_fc(clinical_features)
            combined_features = torch.cat([M, clinical_embedding], dim=1)
        else:
            combined_features = M

        # 特征注意力加权
        attention_weights = self.feature_attention(combined_features)
        weighted_features = combined_features * attention_weights

        # 分类
        logits = self.classifiers(weighted_features)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        results_dict = {}
        if return_features:
            results_dict.update({
                'features': weighted_features,
                'attention_weights': A_raw,
                'clinical_features': clinical_embedding if clinical_features is not None else None
            })

        return logits, Y_prob, Y_hat, A_raw, results_dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class MIL_fc(nn.Module):
    def __init__(self, size_arg="small", dropout=0., n_classes=2, top_k=1, embed_dim=1024, clinical_dim=10):
        super().__init__()
        assert n_classes == 2
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        self.top_k = top_k

        # Image feature branch
        self.image_fc = nn.Sequential(
            nn.Linear(size[0], size[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Clinical feature branch
        self.clinical_fc = nn.Sequential(
            nn.Linear(clinical_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention mechanism for feature fusion
        self.attention = nn.Sequential(
            nn.Linear(size[1] + 256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=0)  # Softmax over instances
        )

        # Classifier
        self.classifier = nn.Linear(size[1] + 256, n_classes)

    def forward(self, h, clinical_features=None, return_features=False):
        # Process image features
        image_feats = self.image_fc(h)  # (N, size[1])

        # Process clinical features
        if clinical_features is not None:
            clinical_feats = self.clinical_fc(clinical_features)  # (1, 256)
            clinical_feats = clinical_feats.repeat(image_feats.shape[0], 1)  # Repeat for each instance
            combined_feats = torch.cat((image_feats, clinical_feats), dim=1)  # (N, size[1] + 256)
        else:
            combined_feats = image_feats

        # Attention mechanism
        attention_weights = self.attention(combined_feats)  # (N, 1)
        weighted_feats = combined_feats * attention_weights  # (N, size[1] + 256)

        # Classification
        logits = self.classifier(weighted_feats)  # (N, n_classes)
        y_probs = F.softmax(logits, dim=1)  # (N, n_classes)

        # Select top-k instances
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1]  # (top_k,)
        top_instance = logits[top_instance_idx]  # (top_k, n_classes)
        Y_hat = torch.topk(top_instance, 1, dim=1)[1]  # (top_k, 1)
        Y_prob = F.softmax(top_instance, dim=1)  # (top_k, n_classes)

        results_dict = {}
        if return_features:
            top_features = weighted_feats[top_instance_idx]  # (top_k, size[1] + 256)
            results_dict.update({'features': top_features})

        return top_instance, Y_prob, Y_hat, y_probs, results_dict


class MIL_fc_mc(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., n_classes = 2, top_k=1, embed_dim=1024):
        super().__init__()
        assert n_classes > 2
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.top_k=top_k
        self.n_classes = n_classes
        assert self.top_k == 1
    
    def forward(self, h, return_features=False):       
        h = self.fc(h)
        logits = self.classifiers(h)

        y_probs = F.softmax(logits, dim = 1)
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]

        Y_hat = top_indices[1]
        Y_prob = y_probs[top_indices[0]]
        
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_indices[0])
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


        

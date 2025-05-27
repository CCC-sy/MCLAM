import sys
# 项目根目录路径（根据实际情况修改，假设你的 models 文件夹在此路径下）
project_root = '/data/run01/scw6f3c/CLAM-master_Clin'
sys.path.append(project_root)
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from models.model_clam import CLAM_SB
import os
import numpy as np

# 假设的参数设置
class Args:
    def __init__(self):
        self.drop_out = 0.25
        self.n_classes = 2
        self.embed_dim = 1024
        self.model_size = "small"
        self.model_type = 'clam_sb'
        self.clinical_dim = 13  # 根据 clin.csv 的列数，除去 slide_id 有 13 个特征
        self.clinical_data_path = '/data/run01/scw6f3c/New_Data/CN/CN_pca_normalized.csv'

args = Args()

# 自定义无标签数据集类，整合临床特征和样本标识
class UnlabeledDataset(Dataset):
    def __init__(self, data_dir, clinical_df, clinical_dim):
        self.data_dir = data_dir
        self.pt_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.clinical_df = clinical_df.set_index('slide_id')
        self.clinical_dim = clinical_dim

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        filename = self.pt_files[idx]
        slide_id = os.path.splitext(filename)[0]  # 获取样本标识
        try:
            clinical_features = self.clinical_df.loc[slide_id].values
        except KeyError:
            print(f"Warning: No clinical data found for slide_id {slide_id}, using zeros.")
            clinical_features = np.zeros(self.clinical_dim)
        data = torch.load(os.path.join(self.data_dir, filename))
        # 返回 slide_id, data, 临床特征 和 默认标签 0
        return slide_id, data, torch.tensor(clinical_features, dtype=torch.float32), 1

# 自定义 collate 函数，处理包含 slide_id 的 batch
def custom_collate_MIL(batch):
    """
    假设 batch 中的每个元素为 (slide_id, data, clinical_features, label)
    """
    # 收集 slide_id（不需要拼接，只作为列表返回）
    slide_ids = [item[0] for item in batch]
    # 对 data 使用 torch.cat 拼接，假设 data 为张量且维度合适
    imgs = torch.cat([item[1] for item in batch], dim=0)
    # 对临床特征进行堆叠
    clinical_features = torch.stack([item[2] for item in batch], dim=0)
    # 对标签进行堆叠
    labels = torch.tensor([item[3] for item in batch])
    return slide_ids, imgs, clinical_features, labels

# 加载临床数据
clinical_df = pd.read_csv(args.clinical_data_path)

# 数据目录
data_root_dir = '/data/run01/scw6f3c/New_Data/CLAM_RESULTS/features'
data_dir = os.path.join(data_root_dir, 'pt_files')

# 创建无标签数据集
unlabeled_dataset = UnlabeledDataset(data_dir, clinical_df, args.clinical_dim)

# 创建数据加载器，直接使用 DataLoader 并设置 shuffle=False 保证顺序一致
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=1, num_workers=1, 
                              shuffle=False, collate_fn=custom_collate_MIL)

# 初始化模型，包含临床特征输入
model_dict = {
    "dropout": args.drop_out,
    'n_classes': args.n_classes,
    "embed_dim": args.embed_dim,
    "size_arg": args.model_size,
    "clinical_dim": args.clinical_dim
}
model = CLAM_SB(**model_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 加载训练好的模型权重
checkpoint_path = '/data/run01/scw6f3c/CLAM-master_Clin/results/7model_clam_sb_reg_1e-5_bag_weight_0.8_lr_5e-4/s_2_checkpoint.pt'
ckpt = torch.load(checkpoint_path, weights_only=False)
ckpt_clean = {}
for key in ckpt.keys():
    if 'instance_loss_fn' in key:
        continue
    ckpt_clean.update({key.replace('.module', ''): ckpt[key]})
model.load_state_dict(ckpt_clean, strict=True)
model.eval()

# 对无标签数据进行分类，同时保留样本标识
predictions = {}
with torch.no_grad():
    for slide_ids, data, clinical_features, _ in unlabeled_loader:
        # 由于 batch_size=1，这里 slide_ids 列表只有一个元素
        slide_id = slide_ids[0]
        data = data.to(device)
        clinical_features = clinical_features.to(device)
        logits, Y_prob, Y_hat, _, _ = model(data, clinical_features)
        predicted_class = Y_hat.item()
        predictions[slide_id] = predicted_class

# 输出预测结果（样本标识和对应预测标签）
print("预测结果:")
for slide_id, pred in predictions.items():
    print(f"slide_id {slide_id}: label {pred}")

# 将 predictions 转换为 DataFrame 并保存为 CSV 文件
df = pd.DataFrame(list(predictions.items()), columns=['slide_id', 'pred'])
output_csv = '/HOME/scw6f3c/run/CLAM-master_Clin/test/predictions/predictions.csv'
df.to_csv(output_csv, index=False)
print(f"预测结果已保存到 {output_csv}")
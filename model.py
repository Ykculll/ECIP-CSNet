import torch
import torch.nn as nn
import torch.nn.functional as Function


# 一维卷积，编码器
class EncoderEEG(nn.Module):
    def __init__(self, channel_dim=62, fea_dim=5, hidden_dim=32):
        super(EncoderEEG, self).__init__()
        
        # Linear层将[62,5]升维到[62,32]
        self.linear = nn.Linear(fea_dim, hidden_dim)
        
        # 一维卷积层

        self.conv1d1 =nn.Sequential(
            nn.Conv1d(in_channels=channel_dim, out_channels=channel_dim, kernel_size=3,stride=2,padding=1),
            nn.BatchNorm1d(channel_dim),
            nn.GELU()
        )
        self.conv1d2 =nn.Sequential(
            nn.Conv1d(in_channels=channel_dim, out_channels=channel_dim, kernel_size=3,stride=2,padding=1),
            nn.BatchNorm1d(channel_dim),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, 62, 5]
        B, C, F = x.shape
        
        # Linear升维 [B, 62, 5] → [B, 62, 32]
        x = self.linear(x)  # [B, 62, 32]
        # conv1d [B, 62, 32] → [B, 62, 32]
        x = self.conv1d1(x)
        x = self.conv1d2(x)
              
        # 返回一维卷积后的特征
        return x

# topk通道筛选
class TopkChannel(nn.Module):
    def __init__(self, k_channels=31, hidden_dim = 32):
        super().__init__()
        self.k = k_channels
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim // 4, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )   

    def forward(self, x):
        # 此处的x为encoder提取的特征
        B, C, F = x.shape
        score = self.scorer(x)  # [B, 62, 1]
        score = score.squeeze(-1)  # [B, C]

        # softmax过后的通道得分，用于K
        kl_std = torch.softmax(score,dim=1)

        # 计算k值
        k = self.k
        
        # 选择绝对值最大的k个通道
        abs_scores = torch.abs(score)
        idx = abs_scores.topk(k, dim=1).indices  # [B, k]
        
        # 获取对应的k个通道数据
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, F)  # [B, k, F]
        x_topk = torch.gather(x, dim=1, index=idx_exp)  # [B, k, F]
        
        return x_topk, idx, kl_std


# 重构 自编码器
class DecoderEEG(nn.Module):
    def __init__(self, channel_dim=62, fea_dim=8, hidden_dim=32):
        super().__init__()
        # 可学习的补齐向量，形状 [1, 1, fea_dim]
        self.fill_vector = nn.Parameter(torch.zeros(1, 1, fea_dim))
        
        # 额外加一层映射回原始特征维度 (5)
        self.to_original = nn.Linear(fea_dim, 5)

        # 自编码器
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=62, out_channels=62, kernel_size=1, padding=0),
            nn.BatchNorm1d(62),
            nn.GELU()
            )

    def forward(self, x_topk, topk_idx, total_channels=62):

        B, k, F = x_topk.shape

        # 复制可学习变量62份
        x_dec = self.fill_vector.repeat(B, total_channels, 1)
        x_dec.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, F), x_topk)
  
        # 映射回原始维度 5
        decoded = self.decoder(x_dec)
        decoded = self.to_original(decoded)  # [B, 62, 5]

        return decoded  # [B, total_channels, F]

# 分类器
class Classifier(nn.Module):
    def __init__(self, fea_dim=8, hidden_dim=64, num_classes=4, k_channels=31):
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(k_channels * fea_dim, 64),    # [B, k*32] → [B, hidden]
            nn.BatchNorm1d(64),
            nn.ReLU(inplace =True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)        #  [B, 4]
        )

    def forward(self, x):
        return self.classifier(x)

# 总网络
class Net(nn.Module):
    def __init__(self, fea_dim=5, channel_dim=62, num_classes=4, topk_ratio=0.5, hidden_dim=32):
        super(Net, self).__init__()
        self.k = int(channel_dim * topk_ratio)
        self.encoder = EncoderEEG(channel_dim=channel_dim, fea_dim=fea_dim, hidden_dim=hidden_dim)
        self.channel_choose = TopkChannel(k_channels=self.k, hidden_dim=32)
        self.decoder = DecoderEEG(channel_dim=channel_dim, fea_dim=hidden_dim // 4)
        self.classifier = Classifier(fea_dim=8, hidden_dim=64, num_classes=4, k_channels=self.k)

    def forward(self, x, mode = 'pretrain'):
        # 编码器：注意力+一维卷积 -> 通道得分 [B,62,1] 和特征 [B,62,32]
        features = self.encoder(x)
        
        # 筛选绝对值最大的k个通道，应用tanh激活和点乘
        x_topk, topk_idx, kl_std= self.channel_choose(features)
        
        if mode == 'pretrain':
            x_recon = self.decoder(x_topk, topk_idx, total_channels = 62)
            return x_recon, kl_std
        elif mode == 'maintrain':
            # 展平送入分类器
            B = x_topk.shape[0]
            x_flat = x_topk.reshape(B, -1)  # [B, k*32]
            pred = self.classifier(x_flat)
            x_recon = self.decoder(x_topk, topk_idx, total_channels = 62)

            return x_recon, pred, kl_std


if __name__ == "__main__":
    import torch
    # 随机生成一个batch的EEG特征数据 [32, 62, 5]
    x = torch.randn(32, 62, 5)
    net = Net(fea_dim=5, channel_dim=62, num_classes=4)
    
    # 前向传播
    pred, topk_scores, topk_idx = net(x)
    print('模型输出:')
    print('pred shape:', pred.shape)        # 应为 [32, 4]
    print('topk_scores shape:', topk_scores.shape)  # 应为 [32, 31]
    print('topk_idx shape:', topk_idx.shape)        # 应为 [32, 31]
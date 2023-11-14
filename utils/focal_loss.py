# 导入相关库
import torch
import torch.nn as nn
import torch.nn.functional as F
# 定义一个Focal loss类
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute the focal loss
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
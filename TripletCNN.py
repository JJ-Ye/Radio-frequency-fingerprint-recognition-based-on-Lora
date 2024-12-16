import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F
import math
from models import MLP,  CNN
import random
from collections import defaultdict
class CombinedLoss(nn.Module):
    def __init__(self, margin, alpha):
        super(CombinedLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.alpha = alpha  # 权重因子

    def forward(self, anchor, positive, negative, logits, labels):
        # 三元损失
        triplet_loss = self.triplet_loss(anchor, positive, negative)
        # 分类交叉熵损失
        cross_entropy_loss = self.cross_entropy_loss(logits, labels)
        # 总损失
        total_loss = self.alpha * triplet_loss + (1 - self.alpha) * cross_entropy_loss
        return total_loss





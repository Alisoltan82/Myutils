import torch
import torch.nn as nn
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.CE = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        minus_logpt = self.CE(input, target)
        pt = (1 - torch.exp(-minus_logpt)).clamp(0.0001)  # clamp to avoid NaNs
        focal_loss = (1 - pt)**self.gamma * minus_logpt

        if self.alpha is not None:
            class_weights = self.alpha[target]
            focal_loss *= class_weights

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()

        return focal_loss
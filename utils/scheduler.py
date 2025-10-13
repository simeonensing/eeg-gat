from __future__ import annotations

import math

import torch



class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, last_epoch: int = -1):
        self.warmup_epochs = max(1, warmup_epochs)
        self.max_epochs = max_epochs
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            return float(epoch + 1) / float(self.warmup_epochs)
        progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

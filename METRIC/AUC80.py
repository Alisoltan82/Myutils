import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric




if __name__ == "__main__":
    class pAUC08torchmetric(Metric):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.add_state("preds", default=[], dist_reduce_fx="cat")
            self.add_state("target", default=[], dist_reduce_fx="cat")

        def update(self, preds, target) -> None:
            self.preds.append(preds)
            self.target.append(target)

        def compute(self):
            # parse inputs
            preds = torch.cat(self.preds)
            target = torch.cat(self.target)
            
            # some intermediate computation...
            sorted_indices = torch.argsort(preds, descending=True)
            
            target_sorted = target[sorted_indices]
            scores_sorted = preds[sorted_indices]

            pos_count = (target_sorted == 1).sum().item()
            neg_count = (target_sorted == 0).sum().item()

            total_pos = pos_count
            tp = 0
            fp = 0
            tpr = 0.0
            fpr = 0.0
            pauc = 0.0

            for i in range(len(target_sorted)):
                if target_sorted[i] == 1:
                    tp += 1
                    tpr = tp / total_pos
                else:
                    fp += 1
                    fpr = fp / neg_count

                    if tpr >= 0.8:
                        pauc += (tpr - 0.8) * fpr
                        min_tpr = tpr
                        if tpr == 1.0:
                            break

            return pauc
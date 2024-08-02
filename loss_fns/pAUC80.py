import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

def pauc_80(y_true, y_scores, min_tpr=0.8):
    sorted_indices = np.argsort(y_scores, descending=True)
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    pos_count = np.sum(y_true_sorted == 1).item()
    neg_count = np.sum(y_true_sorted == 0).item()
    total_pos = pos_count
    
    tp = 0
    fp = 0
    tpr = 0.0
    fpr = 0.0
    pauc = 0.0
    
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
            tpr = tp / total_pos
        else:
            fp += 1
            fpr = fp / neg_count
            
            if tpr >= min_tpr:
                pauc += (tpr - min_tpr) * fpr
                min_tpr = tpr
                if tpr == 1.0:
                    break
    return pauc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn






    
def _common_step(self, batch , batch_idx):
    '''
    A step to be added to all steps in the model
    then tailor your scores from the loss output
    '''
    x,y = batch
    preds = self.forward(x)
    loss = self.loss(preds ,y)
    return loss , preds , y
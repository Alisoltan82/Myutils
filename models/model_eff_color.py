!git clone https://github.com/Alisoltan82/Myutils.git


import torch
from torch import nn , optim, Tensor, manual_seed, argmax
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.io import read_image
import torchmetrics
import torch.nn.functional as F
from Myutils.loss_fns.customm import pAUC08torchmetric


if __name__ == "__main__":

    class SkinModel(pl.LightningModule):
        def __init__(self , lr = 1e-4):
            super().__init__()
            self.model = torchvision.models.efficientnet_v2_m()
            self.model.features[0][0] = torch.nn.Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[1] = torch.nn.Linear(in_features=1280 , out_features=1)
            self.save_hyperparameters()

            #optim
            self.lr = lr
            self.optimizer = torch.optim.Adam(self.model.parameters() , lr = lr)
            self.schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = self.optimizer ,patience = 5 , verbose = True)

            #loss
            self.loss = torch.nn.BCEWithLogitsLoss()

            #Metrics
            self.pauc = pAUC08torchmetric()
            self.train_acc = torchmetrics.Accuracy(task = 'binary' )
            self.val_acc = torchmetrics.Accuracy(task = 'binary')
            self.test_acc = torchmetrics.Accuracy(task = 'binary')
            self.precision = torchmetrics.classification.BinaryPrecision()
            self.recall = torchmetrics.classification.BinaryRecall()
            self.AUC = torchmetrics.classification.BinaryAUROC()

        def forward(self,image):
            
            pred = F.softmax(self.model(image) , dim = 0)
            return pred 


        def training_step(self, batch , batch_idx , config = None):
            x_ray, label  = batch

            label = label.to(torch.float32)
            pred = self(x_ray).to(torch.float32)
            pred = pred.squeeze(1)
    
            loss = self.loss(pred, label)
            accuracy = self.train_acc(pred,label)
            recall = self.recall(pred , label)
            pauc = self.pauc(pred , label)
            AUC = self.AUC(pred , label)
            self.config = wandb.config
        

            self.log_dict({'train_accuracy': accuracy , 'train_loss': loss  , 'recall':recall , 'AUC':AUC , "PAUC":pauc}
                        ,on_step = False ,on_epoch = True , prog_bar = True )

            return loss


        def validation_step(self, batch , batch_idx):
            x_ray, label  = batch
            label = label.to(torch.float32) #torch.Size([32])
                
            pred = self(x_ray )             
            pred = pred.to(torch.float32)
            pred = pred.squeeze(1)          #torch.Size([32])
            
            loss = self.loss(pred, label)
            accuracy = self.val_acc(pred,label)
            val_recall = self.recall(pred,label)
            
            
            self.log_dict({'val_loss': loss ,'val_accuracy': accuracy , 'val_recall': val_recall}
                        ,on_step = False ,on_epoch = True , prog_bar = True)

            return loss



        def test_step(self, batch , batch_idx):
            x_ray, label  = batch
            label = label.to(torch.float32)
            label = label.unsqueeze(1)
            pred = self(x_ray )
            pred = pred.to(torch.float32)
            pred = pred.unsqueeze(1)
            loss = self.loss(pred, label)
            self.log('test_loss', loss , on_epoch = True)


        def configure_optimizers(self):
            return [self.optimizer]
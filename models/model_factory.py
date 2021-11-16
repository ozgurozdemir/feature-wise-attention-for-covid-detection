import numpy as np
import time
import os
import copy

import torch
from torch import nn 
from models.attention_net import *
from models.feature_wise_attention import *


class Model(nn.Module):
    def __init__(self, args):
        """
            End-to-end Network
            
            Args:
                checkpoint_dir : str,   location for saving checkpoints
                load_checkpoint: bool,  loading weights from pretrained network
                is_cuda        : bool,  usage of GPU or CPU
                apply_mixup    : bool,  apply mixup augmentation
                model_type     : str,   backbone arch for feature-wise attention network or 
                                        depth for attention-net network
                model_params   : dict,  hyper-parameters for the network
                epochs         : int,   num of epochs to train the network
                lr             : float, learning rate
                
                
        """
        super(Model, self).__init__()
        self.checkpoint_dir  = args["checkpoint_dir"]
        self.load_checkpoint = args["load_checkpoint"]
        self.model_type      = args["model_type"]
        self.model_params    = args["model_params"]
        self.is_cuda         = args["is_cuda"]
        self.epochs          = args["epochs"]
        self.lr              = args["lr"]
        self.apply_mixup     = args["apply_mixup"]
        self.mixup_alpha = 0.2
        
        if "attention" in self.model_type:
            if self.model_type == "attention56":
                self.model = ResidualAttentionModel_56(self.model_params)
            else: 
                self.model = ResidualAttentionModel_92(self.model_params)
        else: 
            self.model = FeatureAttentionNetwork(self.model_params)
        
        self.device = torch.device("cuda" if self.is_cuda else "cpu")
        self.model         = self.model.to(self.device)
        self.optimizer     = torch.optim.Adam(self.model.parameters(), self.lr)
        self.loss_fn       = torch.nn.CrossEntropyLoss()

    
    # Mixup augmentation src: https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    def mixup(self, inp, tar):
        lamb = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = inp.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lamb * inp + (1 - lamb) * inp[index, :]
        y_a, y_b = tar, tar[index]

        return mixed_x, y_a, y_b, lamb

    def mixup_loss(self, pred, y_a, y_b, lam):
        return lam * self.loss_fn(pred, y_a) + (1 - lam) * self.loss_fn(pred, y_b)
    
    
    def train_step(self, inp, tar):
        self.model.train()
        
        inp = inp.to(self.device)
        tar = tar.to(self.device)
        tar = torch.argmax(tar, dim=1)
        
        if self.apply_mixup:
            inp, tar_a, tar_b, lam = self.mixup(inp, tar)
            output = self.model(inp)
            loss = self.mixup_loss(output, tar_a, tar_b, lam)
        else:
            output = self.model(inp)
            loss   = self.loss_fn(output, tar)    
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data
    
    
    def evaluate(self, valid_ds): 
        self.model.eval()
        with torch.no_grad():
            loss = 0
            for i, (inp, tar) in enumerate(valid_ds):
                inp = inp.to(self.device)
                tar = tar.to(self.device)
                tar = torch.argmax(tar, dim=1)

                output = self.model(inp)
                loss += self.loss_fn(output, tar).data

        return loss / (i+1)

    
    def train(self, train_dataset, valid_dataset, save_model=False):
        best_val_loss = 1000.

        for epoch in range(self.epochs):
            print(f">> Epoch {epoch+1} started...")
            start_time = time.time()
            train_loss = 0; valid_loss = 0
            
            # Training
            self.model.train()
            
            for i, (inp, tar) in enumerate(train_dataset):
                train_loss += self.train_step(inp, tar)
                
            train_loss /= (i+1)
            
            # Evaluation
            self.model.eval()
            valid_loss = self.evaluate(valid_dataset)
            
            # Saving the weights
            if best_val_loss > valid_loss:
                best_val_loss = valid_loss
                best_model = copy.deepcopy(self.model)
                
                if save_model:
                    print(f">> Saving model with {best_val_loss} loss...")
                    torch.save(self.model.state_dict(), self.checkpoint_dir)

            print(f":: Epoch {epoch+1} -> Loss {train_loss/(i+1):.4f}, Val Loss {valid_loss:.4f} -- Best val loss {best_val_loss}")
            print(f":: Time taken {time.time()-start_time} sec...\n")

        return best_model
            

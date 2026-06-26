# Base classes for linear regression model

import torch
from torch import nn

# Class for model
class Model(nn.Module):
    def __init__(self,output_num):
        super().__init__()
        self.net = nn.LazyLinear(output_num)

    def forward(self,x):
        return self.net(x)

    def loss(self,y_hat,y):
        raise NotImplementedError
    
    def training_step(self,batch):
        x,y = batch[0],batch[1]
        y_hat = self(x)
        l = self.loss(y_hat,y)
        return l
    
    def configure_optimizer(self):
        raise NotImplementedError
    
    # Class for data loader
class DataLoader:
    def __init__(self,batch_size):
            self.batch_size = batch_size

    def get_dataloader(self):
        raise NotImplementedError
        
    # Class for trainer
class Trainer:
    def __init__(self,max_epoch):
        self.max_epoch = max_epoch

    def prepare_data(self,data):
        self.dataloader = data.get_dataloader()
        self.num_batches = len(self.dataloader)

    def prepare_model(self,model):
        self.model = model
        model.trainer = self

    def fit(self,model,data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizer()
        for epoch in range(self.max_epoch):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError

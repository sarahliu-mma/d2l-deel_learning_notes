import torch
from torch import nn

# Model Module
# This module contains model, loss function, and optimization method
class model_module(nn.Module):
    def __init__(self,output_num): # Inherit properties from nn.Module, create one layer
        super().__init__()
        self.net = nn.LazyLinear(output_num)

    def forward(self,x): # Pass x through the linear layer
        return self.net(x)

    def loss(self, y_hat, y):
        raise NotImplementedError

    def training_step(self,batch):
        x,y = batch[:-1],batch[-1]
        y_hat = self(x) # Pass x through the forward method in model_module
        l = self.loss(y_hat,y)
        return l

    def configure_optimizer(self):
        raise NotImplementedError


# Data Module
# This module contains dataloader
class data_module():
    def __init__(self, batch_size): # Store batch_size as an instance attribute
        self.batch_size = batch_size

    def get_dataloader(self):
        raise NotImplementedError

# Trainer
# Training process management
class trainer():
    def __init__(self, max_epoch):
        self.max_epoch = max_epoch
    
    def prepare_data(self,data): # The class instance, trainer, has the data built by data_module
        self.dataloader = data.get_dataloader() 
        self.num_batches = len(self.dataloader)

    def prepare_model(self,model): # The class instance, trainer, has the model built by model_module
        self.model = model         # Let the trainer knows the model class instance
        model.trainer = self       # Let the model knows trainer class instance
        
    def fit(self,model,data):      #The training process
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizer()
        for epoch in range(self.max_epoch):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
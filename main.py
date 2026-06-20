{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "from torch import nn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Model Module\n",
    "# This module contains model, loss function, and optimization method\n",
    "class model_module(nn.Module):\n",
    "    def __init__(self,output_num): # Inherit properties from nn.Module, create one layer\n",
    "        super().__init__()\n",
    "        self.net = nn.LazyLinear(output_num)\n",
    "\n",
    "    def forward(self,x): # Pass x through the linear layer\n",
    "        return self.net(x)\n",
    "\n",
    "    def loss(self, y_hat, y):\n",
    "        raise NotImplementedError\n",
    "\n",
    "    def training_step(self,batch):\n",
    "        x,y = batch[:-1],batch[-1]\n",
    "        y_hat = self(x) # Pass x through the forward method in model_module\n",
    "        l = self.loss(y_hat,y)\n",
    "        return l\n",
    "\n",
    "    def configure_optimizer(self):\n",
    "        raise NotImplementedError\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Data Module\n",
    "# This module contains dataloader\n",
    "class data_module():\n",
    "    def __init__(self, batch_size): # Store batch_size as an instance attribute\n",
    "        self.batch_size = batch_size\n",
    "\n",
    "    def get_dataloader(self):\n",
    "        raise NotImplementedError\n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Trainer\n",
    "# Training process management\n",
    "class trainer():\n",
    "    def __init__(self, max_epoch):\n",
    "        self.max_epoch = max_epoch\n",
    "    \n",
    "    def prepare_data(self,data): # The class instance, trainer, has the data built by data_module\n",
    "        self.dataloader = data.get_dataloader() \n",
    "        self.num_batches = len(self.dataloader)\n",
    "\n",
    "    def prepare_model(self,model): # The class instance, trainer, has the model built by model_module\n",
    "        self.model = model         # Let the trainer knows the model class instance\n",
    "        model.trainer = self       # Let the model knows trainer class instance\n",
    "        \n",
    "    def fit(self,model,data):      #The training process\n",
    "        self.prepare_data(data)\n",
    "        self.prepare_model(model)\n",
    "        self.optim = model.configure_optimizer()\n",
    "        for epoch in range(self.max_epoch):\n",
    "            self.fit_epoch()\n",
    "\n",
    "    def fit_epoch(self):\n",
    "        raise NotImplementedError\n",
    "    \n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
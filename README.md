# Deep Learning Framework from Scratch

A minimal deep learning framework built while studying [Dive into Deep Learning (d2l.ai)](https://d2l.ai/index.html).

## Project Structure

├── model_module.py       # Model, loss function, optimizer 

├── data_module.py        # Dataloader

├── trainer.py            # Training process management

└── README.md

## Modules

### model_module
Base class for all models. Inherits from `nn.Module`.
- `forward(x)` — forward pass
- `loss(y_hat, y)` — define loss function in subclass
- `training_step(batch)` — compute loss for one batch
- `configure_optimizer()` — define optimizer in subclass

### data_module
Base class for data loading.
- `get_dataloader()` — define dataloader in subclass

### trainer
Manages the training loop.
- `fit(model, data)` — start training
- `fit_epoch()` — define one epoch's logic in subclass

## Usage

```python
my_model = model_module(output_num=1)
my_data = data_module(batch_size=32)
my_trainer = trainer(max_epoch=10)

my_trainer.fit(my_model, my_data)
```

## Reference
- [Dive into Deep Learning](https://d2l.ai/index.html)

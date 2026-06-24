# D2L Deep Learning Notes

Personal implementations while studying [Dive into Deep Learning (d2l.ai)](https://d2l.ai)

## Structure

```
├── sarahdl_baseclass.py    # Base classes (Model, DataLoader, Trainer)
├── linear_regression_scratch/
│   ├── linear_regression_scratch.ipynb
│   └── linear_regression_application.ipynb
└── TBC/
│   ├── XX_model_scratch.ipynb
│   └── XX_model_application.ipynb
└── README.md
```

## Base Classes

`sarahdl_baseclass.py` contains three base classes used across all implementations:
- `Model` — inherits from `nn.Module`, defines forward, loss, training_step
- `DataLoader` — manages data loading
- `Trainer` — manages the training loop

## Implementations

| Model | Topic | File |
|---------|-------|------|
| Linear Regression | TBC | linear_regression_appication.ipynb |

## Reference
[Dive into Deep Learning](https://d2l.ai)

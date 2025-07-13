# Monitorch

A plug-and-use python module to monitor learning of PyTorch neural networks. Heavily inspired by [an article](https://ai.gopubby.com/better-ways-to-monitor-nns-while-training-7c246867ca4f) by Malcolm Lett. Provides easy to use interface to collect and display

- Loss and custom metrics
- Layer outputs (activation and norms)
- Gradients (norms, scalar products and activation)
- Neural Net's Parameters (norm)

Monitorch manages layer separation, data collection and vizualization with simple exposed methods and classes, so the code is concise and expressive without sacrificing informativness of the vizualizations and broad scope of its application. It also allows user to choose between static matplotlib and dynamic tensorboard plotting to help investigate both small models and keep track of large machines.

# Usage

## Requirments

- python>=3.10
- torch>=2.0.0
- matplotlib>=3.10.0
- tensorboard>=2.19.0

## Installation

Install the module using pip dependency manager.

```{bash}
pip install monitorch
```

## Code

Use `PyTorchInspector` from `monitorch.inspector` to hook your module and lenses (for example `LossMetrics` or `ParameterGradientGeometry`) from `monitorch.lens` to define vizualizations.

```{python}
import torch

from monitorch.inspector import PyTorchInspector
from monitorch.lens import LossMetrics, ParameterGradientGeometry

mynet = MyNeuralNet() # torch.nn.Module subclass
loss_fn = MSELoss()
optimizer = torch.optim.Adam(module.parameters())

inspector = PyTorchInspector(
    lenses = [
        LossMetrics(loss_fn=loss_fn),
        ParameterGradientGeometry()
    ],
    module = mynet,
    vizualizer = "matplotlib"
)

for epoch in range(n_epochs):
    # No changes to your training loop
    # Passes through training and validation datasets remain the same

    ...

    # at the end of an epoch inspector must be ticked
    inspector.tick_epoch()

inspector.vizualizer.show_fig()
```

You can choose other vizualizers by passing `"tensorboard"`, `"print"` or an instance of vizualizer's class from `monitorch.vizualizers`. Note that matplotlib vizualier requires `show_fig()` call to plot.

Currently module supports gradient and parameter collection for arbitrary PyTorch module and output collection for single output architectures (feedforward, convolution, non-transformer autoencoders etc).


## Tests

Tests can be run with `pytest` from root project directory. Lens test have no assertions or other critical functionality tests, but are rather smoke tests to catch unhandled exceptions. To run functionality tests run `pytest -k "not smoke"`.

# TODO

- [ ] Documentation
    - [ ] Setup sphinx
    - [ ] Write docstrings
    - [ ] Create demo notebooks
- [ ] Recurrent Neural Networks output collection
- [ ] Transformer output collection (value vector vs state vector)
- [ ] Debug Lens

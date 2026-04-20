# Monitorch

A plug-and-use python module to monitor learning of PyTorch neural networks. Provides easy to use interface to collect and display

- Loss and custom metrics
- Layer outputs (activation and norms)
- Gradients (norms, oscilations and activation)
- Neural Net's Parameters Evolution

Monitorch manages layer separation, data collection and vizualization with simple exposed methods and classes, so the code is concise and expressive without sacrificing informativness of the visualizations and broad scope of its application. It also allows user to choose between static matplotlib and dynamic tensorboard plotting to help investigate both small models and keep track of large machines.

# Documentation

Documentation can be found here:

[https://monitorch.readthedocs.io/en/latest/](https://monitorch.readthedocs.io/en/latest/).

# Usage

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

## Requirments

- python>=3.10
- torch>=2.0.0

Optional:

- matplotlib>=3.10.0
- tensorboard>=2.19.0

Development:

- sphinx>=8.1.3
- pydata-sphinx-theme>=0.16.1
- pytest>=9.0.3

## Tests

Tests can be run with `pytest` from root project directory. Lens test have no assertions or other critical functionality tests, but are rather smoke tests to catch unhandled exceptions. To run functionality tests run `pytest -k "not smoke"`.

# Other repositories

## Case studies

- [nanochat-monitorch](https://github.com/ZhigaMason/nanochat-monitorch) Karpathy's nanochat with viewed using monitorch.
- [monitorch-experments](https://github.com/ZhigaMason/monitorch-experiments) Repository with my results using monitorch (including nanochat logs).

## Tensorflow

Malcolm Lett has created an excellent Tensorflow library with similar features. So if you prefer Tensorflow, please check it out:

[training-instrumentation-toolkit](https://github.com/malcolmlett/training-instrumentation-toolkit)

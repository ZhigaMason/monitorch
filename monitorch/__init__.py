"""
Monitorch
Visualize and monitor neural network training internals.

This package provides tools for inspecting gradients, weights, and forward/backward
passes in PyTorch models. It is designed to integrate smoothly with TensorBoard and
other visualization tools.

Submodules
----------
inspector   : Interfaces for integrating with PyTorch training loops.
lens        : Lenses that define how raw model information is interpreted.
numerical   : Utilities for statistical tracking (e.g., running mean/var).
vizualizer  : Plotting utilities (under development).
gatherer    : Tools for capturing model internals via hooks.
preprocessor: Data aggregation classes that connect gatherers and inspector.

Examples
--------
>>> from monitorch import PyTorchInspector
>>> inspector = PyTorchInspector(...)
>>> inspector.attach(model)
"""

__version__ = "0.0.2.0"

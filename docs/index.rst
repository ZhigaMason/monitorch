.. Monitorch documentation master file, created by
   sphinx-quickstart on Tue Jul 29 17:39:07 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Monitorch documentation
=======================

A modular tool to inspect, log, and visualize neural network internals during training in PyTorch.
Most of the statistics collected and displayed by this module are described in `three beatiful articles <https://ai.gopubby.com/better-ways-to-monitor-nns-while-training-7c246867ca4f>`_ by Malcolm Lett.

.. container:: columns

    .. container:: column-left

        **Quickstart**

        To install monitorch run

        ::

                pip install monitorch

        in your virtual environment.

        To use monitorch it is enough to define an inspector as shown below

        ::

                from monitorch.inspector import PyTorchInspector
                from monitorch.lens import (
                        LossMetrics,
                        ParameterGradientGeometry,
                        OutputActivation
                )

                loss_fn = nn.NLLLoss() # Any loss

                inspector = PyTorchInspector(
                        lenses = [
                                LossMetrics(loss_fn=loss_fn),
                                ParameterGradientGeometry(),
                                OutputActivation()
                        ]
                )

        And to attach the inspector to a net that will be trained. At an end of an epoch or episode inspector must be ticked.

        ::

                inspector.attach(custom_net)

                for epoch in range(N_EPOCHS):
                        ...
                        # Training and validation
                        # subloops remain the same
                        ...
                        inspector.tick_epoch()

        Lastly, if visualizer is set to ``"matplotlib"`` (default), figure must be shown.

        ::

                fig = inspector.visualizer.show_fig()

        Now you can see the training process in great detail!

        For further examples see demonstration notebooks.

        .. toctree::
                :maxdepth: 2

                notebooks/title_page

    .. container:: column-right

        .. toctree::
                :maxdepth: 3

                api/index

**Author**: Maksym Khavil
**Repository**: https://github.com/ZhigaMason/monitorch

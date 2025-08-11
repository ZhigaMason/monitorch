from torch.nn import Module
from typing_extensions import Self

from monitorch.lens import AbstractLens
from monitorch.preprocessor import ExplicitCall
from monitorch.visualizer import _vizualizer_dict, AbstractVisualizer, MatplotlibVisualizer

class PyTorchInspector:
    """
    One class to rule them all.

    ``PyTorchInspector`` is a class that manages interactions between lenses, visualizers and user defined module.

    To use inspector one needs to initialize the inspector and provide it the module to monitor.
    During initialization lenses from :mod:`monitorch.lens` must be provided.
    The only thing that is required during training is to call :meth:`tick_epoch` on the end of each epoch.
    Optionally one could push additional metrics using :meth:`push_metric` and :meth:`push_loss`.

    If visualizer is ``'matplotlib'``, then ``'show_fig()'`` must be called on :attr:`visualizer`,
    otherwise the plot will be drawn during training.

    Parameters
    ----------
    lenses : list[AbstractLens]
        List of objects from :mod:`monitorch.lens`, used to collect and plot data.

    visualizer : str|AbstractVisualizer = 'matplotlib'
        Visualizer to draw plots, must be either a visualizer object from :mod:`monitorch.visualizer`
        or a string ``'matplotlib'``, ``'tensorboard'`` or ``'print'``.
    module : None|torch.nn.Module = None
        Optional neural network to examine, can be added later using :meth:`attach`.

    depth : int = -1
        Depth to unfold neural net injection tree. For example ``depth=0`` returns the model itself,
        ``depth=1`` returns modules directly contained in ``module`` object. Default is ``depth=-1``,
        that is to unfold until leaf modules are reached.
    module_name_prefix : str = '.'
        Delimiter to separate names of parent and child modules.

    train_loss_str = 'train_loss'
        String to be used for training loss.
    non_train_loss_str = 'val_loss
        String to be used for validation/testing/development loss.

    Attributes
    ----------
    lenses : list[AbstractLens]
        List of objects from :mod:`monitorch.lens`, used to collect and plot data.
        Exatcly the same object as the one provided during initialization.

    visualizer : AbstractVisualizer
        Visualizaer object that draws all plots. Can be hot-swapped.

    attached : bool
        Flag indicating if inspector is attached.

    epoch_counter : int
        Internal epoch counter used in :meth:`tick_epoch`, increments on call.

    depth : int
        Depth to unfold module inclusion tree.
    module_name_prefix : str
        Delimiter to separate names of parent and child modules.

    Examples
    --------

    Basic usage with ``'LossMetrics'``, ``'OutputActivation'`` and ``'ParameterGradientGeometry'``
    may look something like this.

    >>> from monitorch.inspector import PyTorchInspector
    >>> from monitorch.lens import LossMetrics, OutputActivation, ParameterGradientGeometry
    >>> 
    >>> loss_fn = nn.NLLLoss()
    >>> 
    >>> inspector = PyTorchInspector(
    ...     lenses = [
    ...         LossMetrics(loss_fn = loss_fn),
    ...         OutputActivation(),
    ...         ParameterGradientGeometry()
    ...     ],
    ...     module = mynet,
    ...     visualizer='matplotlib'
    ... )
    >>> 
    >>> for epoch in range(N_EPOCHS):
    ...     for data, label in train_dataloader:
    ...         optimizer.zero_grad()
    ...         prediction = mynet(data)
    ...         loss = loss_fn(prediction, label)
    ...         loss.backward()
    ...         optimizer.step()
    ... 
    ...     with torch.no_grad(): # outputs inside this block are not recorded
    ...         for data, label in val_dataloader:
    ...             prediction = mynet(data)
    ...             loss = loss_fn(prediction, label)
    ... 
    ...     inspector.tick_epoch() # ticking the epoch
    >>> 
    >>> inspector.visualizer.show_fig()
    """

    def __init__(
            self,
            lenses : list[AbstractLens], *,
            visualizer : str|AbstractVisualizer = 'matplotlib',
            module : None|Module = None,
            depth : int = -1,
            module_name_prefix : str = '.',
            train_loss_str = 'train_loss',
            non_train_loss_str = 'val_loss'
    ):
        self.lenses = lenses
        self._call_preprocessor = ExplicitCall(train_loss_str, non_train_loss_str)
        self.depth = depth
        self.module_name_prefix = module_name_prefix
        self.attached = False

        self.epoch_counter = 0

        if isinstance(visualizer, str):
            if visualizer not in _vizualizer_dict:
                raise AttributeError(f"Unknown vizualizer, string defined vizualizer must be one of {list(_vizualizer_dict.keys())} ")
            self.visualizer = _vizualizer_dict[visualizer]()
        else:
            self.visualizer : AbstractVisualizer = visualizer

        for lens in self.lenses:
            lens.register_foreign_preprocessor(self._call_preprocessor)
            lens.introduce_tags(self.visualizer)
        if module is not None:
            self.attach(module)

    def attach(self, module : Module) -> Self:
        """
        Attaches inspector to a module.

        Unfolds inclusion module tree guided by ``depth`` set during initialization.
        Registers submodules onto every lens.

        Parameters
        ----------
        module : torch.nn.Module
            Neural net to attach to.

        Returns
        -------
        Self
            Builder pattern.
        """
        if self.attached:
            self.detach()
        module_names = PyTorchInspector._module_leaves(module, self.depth, self.module_name_prefix)
        for module, name in module_names:
            for lens in self.lenses:
                lens.register_module(module, name)
        self.attached = True
        return self

    def detach(self) -> Self:
        """
        Detaches all lenses from modules.

        Returns
        -------
        Self
            Builder pattern.
        """
        assert self.attached, "Inspector must be attached to module before detaching"
        self.epoch_counter = 0
        self._call_preprocessor.reset()
        for lens in self.lenses:
            lens.detach_from_module()
        if isinstance(self.visualizer, MatplotlibVisualizer):
            self.visualizer.reset_fig()
        self.attached = False
        return self

    def push_metric(self, name : str, value : float, *, running : bool=True):
        """
        Pushes metric, that can be accessed by :class:`monitorch.lens.LossMetrics`.

        Parameters
        ----------
        name : str
            Name of the metric to save.
        value : float
            Metric's value.
        running : bool = True
            Flag indicating if metric should be saved in-place (True) or in-memory (False).
        """
        if running:
            self._call_preprocessor.push_running(name, value)
        else:
            self._call_preprocessor.push_memory(name, value)

    def push_loss(self, value : float, *, train : bool, running : bool = True):
        """
        Pushes loss, that can be accessed by :class:`monitorch.lens.LossMetrics`.

        Parameters
        ----------
        value : float
            Loss value.
        train : bool
            Flag indicating if it is training loss.
        running : bool = True
            Flag indicating if metric should be saved in-place (True) or in-memory (False).
        """
        self._call_preprocessor.push_loss(value, train=train, running=running)

    def tick_epoch(self, epoch : int|None=None):
        """
        Ticks epoch to postprocess data and draw plots.

        Parameters
        ----------
        epoch : int|None = None
            Optional epoch counter, default is incremental :attr:`epoch_counter`.
        """
        if epoch is not None:
            self.epoch_counter = epoch
        for lens in self.lenses:
            lens.finalize_epoch()
            lens.vizualize(self.visualizer, self.epoch_counter)
            lens.reset_epoch()
        self._call_preprocessor.reset()
        self.epoch_counter += 1

    @staticmethod
    def _decide_prefix(prefix : str, grand_name : str):
        """ Utility function for depth=0 name composition. """
        return prefix if grand_name else ''

    @staticmethod
    def _module_leaves(module : Module, depth : int = -1, prefix : str = '.') -> list[tuple[Module, str]]:
        """
        A function to extract nodes at defined depth from module inclusion tree.
        If ``depth=-1`` calls :meth:`_module_deep_leaves`,
        otherwise recursively goes down the tree decreasing depth.

        Parameters
        ----------
        module : torch.nn.Module
            Module which inclusion tree must be unfolded.
        depth : int = -1
            Depth to which the module must be unfolded, default is -1, i.e., until leaf nodes.
        prefix : str = '.'
            Delimiter to separate names of parent and child modules.

        Returns
        -------
        list[tuple[Module, str]]
            List of module object and their path name.
        """
        assert depth >= -1, "Depth of leaves must be non-negative or -1 (maximal depth)"
        if depth == -1:
            return PyTorchInspector._module_deep_leaves(module, prefix=prefix)
        if depth == 0:
            return [(module, '')]

        ret = []
        for name, child in module.named_children():
            ret += [(module, name + PyTorchInspector._decide_prefix(prefix, grand_name) + grand_name) for module, grand_name in PyTorchInspector._module_leaves(child, depth - 1)]
        return ret

    @staticmethod
    def _module_deep_leaves(module : Module, prefix : str) -> list[tuple[Module, str]]:
        """
        A function to extract leaves from module inclusion tree.

        The function is recursive.

        Parameters
        ----------
        module : torch.nn.Module
            Module which inclusion tree must be unfolded.
        prefix : str = '.'
            Delimiter to separate names of parent and child modules.

        Returns
        -------
        list[tuple[Module, str]]
            List of module object and their path name.
        """
        ret = []
        for name, child in module.named_children():
            ret += [(module, name + PyTorchInspector._decide_prefix(prefix, grand_name) + grand_name) for module, grand_name in PyTorchInspector._module_deep_leaves(child, prefix=prefix)]
        if ret == []:
            return [(module, '')]
        return ret

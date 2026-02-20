from collections.abc import Callable, Iterable

from torch.nn import Module
from typing_extensions import Self

from monitorch.lens import AbstractLens
from monitorch.preprocessor import ExplicitCall
from monitorch.visualizer import AbstractVisualizer, MatplotlibVisualizer, _vizualizer_dict

from .inspector_state import InspectorState


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

    is_active_fn : int | Callable[[int], bool] = 1
        Function deciding if inspector is active (collects and visualizes data) for given epoch. Passed directly to `InspectorState`. Integer values correspond to  function ``epoch % n == 0``, where `n` is passed value.


    Attributes
    ----------
    lenses : list[AbstractLens]
        List of objects from :mod:`monitorch.lens`, used to collect and plot data.
        Exatcly the same object as the one provided during initialization.

    visualizer : AbstractVisualizer
        Visualizaer object that draws all plots. Can be hot-swapped.

    state : InspectorState
        State object representing inspectors inner state. Weak-referenced by gatherers.

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
        lenses: list[AbstractLens],
        *,
        visualizer: str | AbstractVisualizer = 'matplotlib',
        module: None | Module = None,
        depth: int = -1,
        module_name_prefix: str = '.',
        train_loss_str='train_loss',
        non_train_loss_str='val_loss',
        is_active_fn: int | Callable[[int], bool] = 1,
    ):
        self.lenses = lenses
        self._call_preprocessor = ExplicitCall(train_loss_str, non_train_loss_str)
        self.depth = depth
        self.module_name_prefix = module_name_prefix
        self.state: InspectorState = InspectorState(is_active_fn=is_active_fn)

        if isinstance(visualizer, str):
            if visualizer not in _vizualizer_dict:
                raise AttributeError(f'Unknown vizualizer, string defined vizualizer must be one of {list(_vizualizer_dict.keys())} ')
            self.visualizer = _vizualizer_dict[visualizer]()
        else:
            self.visualizer: AbstractVisualizer = visualizer

        for lens in self.lenses:
            lens.register_foreign_preprocessor(self._call_preprocessor, self.state)
            lens.introduce_tags(self.visualizer)
        if module is not None:
            self.attach(module)

    def attach(self, module: Module) -> Self:
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
        if self.state.attached:
            self.detach()
        leaf_module_names, non_leaf_module_names = PyTorchInspector._traverse_module_inclusion_tree(module, self.depth, self.module_name_prefix)

        for module, name in leaf_module_names:
            for lens in self.lenses:
                lens.register_leaf_module(module, name, self.state)

        for module, name in non_leaf_module_names:
            for lens in self.lenses:
                lens.register_non_leaf_module(module, name, self.state)

        self.state.attached = True
        return self

    def detach(self) -> Self:
        """
        Detaches all lenses from modules.

        Returns
        -------
        Self
            Builder pattern.
        """
        assert self.state.attached, 'Inspector must be attached to module before detaching'
        self.state.counter = 0
        self._call_preprocessor.reset()
        for lens in self.lenses:
            lens.detach_from_module()
        if isinstance(self.visualizer, MatplotlibVisualizer):
            self.visualizer.reset_fig()
        self.state.attached = False
        return self

    def push_metric(self, name: str, value: float, *, running: bool = True):
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

    def push_loss(self, value: float, *, train: bool, running: bool = True):
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

    def tick_epoch(self, epoch: int | None = None):
        """
        Ticks to postprocess data and draw plots.

        Parameters
        ----------
        epoch : int|None = None
            Optional epoch counter, default ticks :attr:`state`, thus incrementing `counter`.
        """
        if not self.state.is_active:
            if epoch is not None:
                self.state.counter = epoch
            else:
                self.state.tick()
            return

        for lens in self.lenses:
            lens.finalize_epoch()
            lens.vizualize(self.visualizer, self.state.counter)
            lens.reset_epoch()
        self._call_preprocessor.reset()

        if epoch is not None:
            self.state.counter = epoch
        else:
            self.state.tick()

    tick = tick_epoch

    def iter(self, iterable: Iterable) -> Iterable:
        dotick = False
        for x in iterable:
            if dotick:
                self.tick()
            else:
                dotick = True
            yield x
        self.tick()

    def range(self, *args, **kwargs) -> Iterable:
        return self.iter(range(*args, **kwargs))

    @staticmethod
    def _decide_prefix(prefix: str, grand_name: str):
        """Utility function for depth=0 name composition."""
        return prefix if grand_name else ''

    @staticmethod
    def _traverse_module_inclusion_tree(module: Module, depth: int = -1, prefix: str = '.') -> tuple[list[tuple[Module, str]], list[tuple[Module, str]]]:
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
        tuple[list[tuple[Module, str], list[tuple[Module, str]]]
            Lists of leaf (1st value) and non-leaf (2nd value) module object and their path name.
        """
        assert depth >= -1, 'Depth of leaves must be non-negative or -1 (maximal depth)'
        if depth == -1:
            return PyTorchInspector._module_deep_leaves(module, prefix=prefix)
        if depth == 0:
            return [(module, '')], []

        leaves = []
        non_leaves = []
        for name, child in module.named_children():
            child_leaves, child_non_leaves = PyTorchInspector._traverse_module_inclusion_tree(child, depth - 1)
            leaves += [(module, name + PyTorchInspector._decide_prefix(prefix, grand_name) + grand_name) for module, grand_name in child_leaves]
            non_leaves += [(module, name + PyTorchInspector._decide_prefix(prefix, grand_name) + grand_name) for module, grand_name in child_non_leaves]
        if len(leaves) > 0:
            non_leaves.append((module, ''))
        return leaves, non_leaves

    @staticmethod
    def _module_deep_leaves(module: Module, prefix: str) -> tuple[list[tuple[Module, str]], list[tuple[Module, str]]]:
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
        tuple[list[tuple[Module, str], list[tuple[Module, str]]]
            Lists of leaf (1st value) and non-leaf (2nd value) module object and their path name.
        """
        leaves = []
        non_leaves = []
        for name, child in module.named_children():
            child_leaves, child_non_leaves = PyTorchInspector._module_deep_leaves(child, prefix=prefix)
            leaves += [(child_module, name + PyTorchInspector._decide_prefix(prefix, grand_name) + grand_name) for child_module, grand_name in child_leaves]
            non_leaves += [(child_module, name + PyTorchInspector._decide_prefix(prefix, grand_name) + grand_name) for child_module, grand_name in child_non_leaves]
        if len(leaves) == 0:
            leaves = [(module, '')]
        else:
            non_leaves.append((module, ''))
        return leaves, non_leaves

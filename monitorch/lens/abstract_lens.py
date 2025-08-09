from abc import ABC, abstractmethod
from torch.nn import Module
from monitorch.preprocessor import AbstractPreprocessor
from monitorch.visualizer import AbstractVisualizer


class AbstractLens(ABC):
    """
    Base class for all lenses.

    Defines minimal interface that a lens must satisfy to interact with
    :class:`monitorch.inspector.PyTorchInspector` and visualizers from :mod:`monitorch.visualizer`.

    Lens implementation should allocate and manage all gatherers and preprocessors alone.
    """

    @abstractmethod
    def register_module(self, module : Module, module_name : str):
        """
        Registers (or ignores) module.

        Register module, i.e., create and link gatherers.
        The lens should ignore the modules it does not interact with not to overcrowd plots with useless information.

        Parameters
        ----------
        module : torch.nn.Module
            The module object to hook gatherers onto.
        module_name : str
            Name of the module, module's information will be passed to visaulizer under this name.
        """
        pass

    @abstractmethod
    def detach_from_module(self):
        """
        Detaches lens from module.

        Detaches gatherers and resets inner state.
        """
        pass

    @abstractmethod
    def register_foreign_preprocessor(self, ext_ppr : AbstractPreprocessor):
        """
        Registers preprocessor allocated and managed by external environment.

        Gets a week reference to a preprocessor and decides what to do with it.
        Primary example is :class:`monitorch.preprocessor.ExplicitCall`,
        it is usually allocated inside inspector.

        Parameters
        ----------
        ext_ppr : AbstractPreprocessor
            External preprocessor to register (or ignore).
        """
        pass

    @abstractmethod
    def introduce_tags(self, visualizer : AbstractVisualizer):
        """
        Introduces lens's plots to visualizer.

        It is a preparation method for the visualizer.
        Lens should pass all tags to :meth:`register_tags` with appropriate attributes.

        Parameters
        ----------
        visualzier : AbstractVisualizer
            A visualizer object to pass tag attributes to.
        """
        pass

    @abstractmethod
    def finalize_epoch(self):
        """
        Finaizes computations done thorugh epoch.

        During finalization data from preprocessors can be transfered to
        dedicated storages and reorganized in visualizer friendly way.
        """
        pass

    @abstractmethod
    def vizualize(self, visualizer : AbstractVisualizer, epoch : int):
        """
        Passes computed data to visualizer.

        Uses plot method from visualizer to pass the data.

        Parameters
        ----------
        visualizer : AbstractVisualizer
            The visualizer object responsbile for drawing plots.
        epoch : int
            Computation's epoch number.
        """
        pass

    @abstractmethod
    def reset_epoch(self):
        """
        Resets inner state.

        Resets data computed during last epoch and resets preprocessors.
        """
        pass

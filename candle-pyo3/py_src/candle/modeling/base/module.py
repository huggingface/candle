from candle import Tensor, QTensor
from typing import Dict, Tuple, Any, Optional, Union, Iterator, Set
from collections import OrderedDict

TensorLike = Union[Tensor, QTensor]


# see: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py
class Module:
    """
    Pytorch like Module.

    Base class for all neural network modules.

    Your models should also subclass this class.
    """

    _modules: Dict[str, Optional["Module"]]
    _buffers: Dict[str, Optional[TensorLike]]

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes internal Module state
        """
        super().__setattr__("_modules", OrderedDict())
        super().__setattr__("_buffers", OrderedDict())

    def __call__(self, *input):
        """
        Call self as a function.
        """
        return self.forward(*input)

    def forward(self, *input):
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        pass

    def children(self) -> Iterator["Module"]:
        r"""Returns an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        r"""Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.

        Yields:
            (str, Module): Tuple containing a name and child module

        Example::

            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)

        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        r"""Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (str): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{str(module)} is not a Module subclass")
        elif not isinstance(name, str):
            raise TypeError(f"module name should be a string. Got {name}")
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        elif "." in name:
            raise KeyError(f'module name can\'t contain ".", got: {name}')
        elif name == "":
            raise KeyError('module name can\'t be empty string ""')
        self._modules[name] = module

    def register_module(self, name: str, module: Optional["Module"]) -> None:
        r"""Alias for :func:`add_module`."""
        self.add_module(name, module)

    def modules(self) -> Iterator["Module"]:
        r"""Returns an iterator over all modules in the network."""
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self,
        memo: Optional[Set["Module"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (str, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.
        """

        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    yield m

    def buffers(self, recurse: bool = True) -> Iterator[TensorLike]:
        """
        Returns an iterator over module buffers.
        """
        for name, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, TensorLike]]:
        r"""Returns an iterator over module buffers, yielding both the
        name of the buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool, optional): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module. Defaults to True.
            remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

        Yields:
            (str, Tensor): Tuple containing the name and buffer

        Example::

            >>> for name, buf in self.named_buffers():
            >>>     if name in ['running_var']:
            >>>         print(buf.size())

        """
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def _named_members(
        self, get_members_fn, prefix="", recurse=True, remove_duplicate: bool = True
    ):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = (
            self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
            if recurse
            else [(prefix, self)]
        )
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                if remove_duplicate:
                    memo.add(v)
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, v

    def _get_name(self):
        return self.__class__.__name__

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, Module):
            self._modules[__name] = __value
        elif isinstance(__value, (Tensor, QTensor)):
            self._buffers[__name] = __value
        else:
            super().__setattr__(__name, __value)

    def __getattr__(self, __name: str) -> Any:
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if __name in modules:
                return modules[__name]
        if "_buffers" in self.__dict__:
            tensors = self.__dict__["_buffers"]
            if __name in tensors:
                return tensors[__name]
        return super().__getattribute__(__name)

    def __delattr__(self, name):
        if name in self._buffers:
            del self._buffers[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

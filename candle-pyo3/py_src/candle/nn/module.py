from candle import Tensor, QTensor
from typing import (
    Dict,
    Tuple,
    Any,
    Optional,
    Union,
    Iterator,
    Set,
    overload,
    Mapping,
    TypeVar,
    List,
)
from collections import OrderedDict, namedtuple

TensorLike = Union[Tensor, QTensor]


class _IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__


# see: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py
class Module:
    """
    Pytorch like Module.

    Base class for all neural network modules.

    Your models should also subclass this class.
    """

    _modules: Dict[str, Optional["Module"]]
    _buffers: Dict[str, Optional[TensorLike]]
    _non_persistent_buffers_set: Set[str]
    _version: int = 1

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes internal Module state
        """
        super().__setattr__("_modules", OrderedDict())
        super().__setattr__("_buffers", OrderedDict())
        super().__setattr__("_non_persistent_buffers_set", set())

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

    # The user can pass an optional arbitrary mappable object to `state_dict`, in which case `state_dict` returns
    # back that same object. But if they pass nothing, an `OrderedDict` is created and returned.
    T_destination = TypeVar("T_destination", bound=Dict[str, Any])

    @overload
    def state_dict(
        self, *, destination: T_destination, prefix: str = ..., keep_vars: bool = ...
    ) -> T_destination:
        ...

    @overload
    def state_dict(self, *, prefix: str = ..., keep_vars: bool = ...) -> Dict[str, Any]:
        ...

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        r"""Returns a dictionary containing references to the whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.
        Parameters and buffers set to ``None`` are not included.

        .. note::
            The returned object is a shallow copy. It contains references
            to the module's parameters and buffers.

        .. warning::
            Currently ``state_dict()`` also accepts positional arguments for
            ``destination``, ``prefix`` and ``keep_vars`` in order. However,
            this is being deprecated and keyword arguments will be enforced in
            future releases.

        .. warning::
            Please avoid the use of argument ``destination`` as it is not
            designed for end-users.

        Args:
            destination (dict, optional): If provided, the state of module will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            prefix (str, optional): a prefix added to parameter and buffer
                names to compose the keys in state_dict. Default: ``''``.
            keep_vars (bool, optional): by default the :class:`~candle.Tensor` s
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> module.state_dict().keys()
            ['bias', 'weight']

        """

        # TODO: Remove `args` and the parsing logic when BC allows.
        if len(args) > 0:
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == "":
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(
                    destination=destination,
                    prefix=prefix + name + ".",
                    keep_vars=keep_vars,
                )
        return destination

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                if isinstance(buf, Tensor):
                    destination[prefix + name] = buf if keep_vars else buf.detach()
                else:
                    destination[prefix + name] = buf

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        .. warning::
            If :attr:`assign` is ``True`` the optimizer must be created after
            the call to :attr:`load_state_dict`.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
            assign (bool, optional): whether to assign items in the state
                dictionary to their corresponding keys in the module instead
                of copying them inplace into the module's current parameters and buffers.
                When ``False``, the properties of the tensors in the current
                module are preserved while when ``True``, the properties of the
                Tensors in the state dict are preserved.
                Default: ``False``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys

        Note:
            If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"Expected state_dict to be dict-like, got {type(state_dict)}."
            )

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = OrderedDict(state_dict)
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, local_state_dict, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            if assign:
                local_metadata["assign_to_params_buffers"] = assign
            module._load_from_state_dict(
                local_state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + "."
                    child_state_dict = {
                        k: v
                        for k, v in local_state_dict.items()
                        if k.startswith(child_prefix)
                    }
                    load(child, child_state_dict, child_prefix)

        load(self, state_dict)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join(f'"{k}"' for k in unexpected_keys)
                    ),
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0,
                    "Missing key(s) in state_dict: {}. ".format(
                        ", ".join(f'"{k}"' for k in missing_keys)
                    ),
                )

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.
        Additionally, :attr:`local_metadata` can also contain the key
        `assign_to_params_buffers` that indicates whether keys should be
        assigned their corresponding tensor in the state_dict.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        persistent_buffers = {
            k: v
            for k, v in self._buffers.items()
            if k not in self._non_persistent_buffers_set
        }
        local_name_params = persistent_buffers.items()
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if not isinstance(input_param, (Tensor, QTensor)):
                    error_msgs.append(
                        f'While copying the parameter named "{key}", '
                        "expected Tensor-like object from checkpoint but "
                        f"received {type(input_param)}"
                    )
                    continue

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append(
                        "size mismatch for {}: copying a param with shape {} from checkpoint, "
                        "the shape in current model is {}.".format(
                            key, input_param.shape, param.shape
                        )
                    )
                    continue

                try:
                    # Shape checks are already done above -> Just assign tensor
                    setattr(self, name, input_param)
                except Exception as ex:
                    error_msgs.append(
                        f'While copying the parameter named "{key}", '
                        f"whose dimensions in the model are {param.shape} and "
                        f"whose dimensions in the checkpoint are {input_param.shape}, "
                        f"an exception occurred : {ex.args}."
                    )
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix) :]
                    input_name = input_name.split(".", 1)[
                        0
                    ]  # get the name of param/buffer/child
                    if (
                        input_name not in self._modules
                        and input_name not in local_state
                    ):
                        unexpected_keys.append(key)

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

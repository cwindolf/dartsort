from typing import Any
from weakref import proxy, CallableProxyType, ReferenceType
from torch import Tensor
from torch.nn import Module


class BModule(Module):
    """This only exists to silence some pyright messages

    And to extend torch behavior to keep some tensors on CPU without
    stressing too much.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cpu_buffers = {}
        self.b = BufGetter(proxy(self))

    def register_cpu_buffer(self, name: str, buf: Tensor | None):
        self._cpu_buffers[name] = buf

    def register_buffer_or_none(
        self, name: str, buf: Tensor | None, on_device: bool = True
    ):
        if buf is None:
            setattr(self, name, None)
            self.register_cpu_buffer(name, None)
        elif on_device:
            self.register_buffer(name, buf)
        else:
            self.register_cpu_buffer(name, buf)


class BufGetter:
    def __init__(self, module: "CallableProxyType[BModule]"):
        self.module = module

    def __getattr__(self, key) -> Tensor:
        if key in self.module._cpu_buffers:
            return self.module._cpu_buffers[key]
        return self.module.get_buffer(key)

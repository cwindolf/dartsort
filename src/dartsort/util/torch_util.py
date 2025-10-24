from weakref import proxy
from torch import Tensor
from torch.nn import Module


class BModule(Module):
    """This only exists to silence some pyright messages."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.b = BufGetter(proxy(self))


class BufGetter:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, key) -> Tensor:
        return self.module.get_buffer(key)

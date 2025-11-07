from torch import Tensor
from torch.nn import Module


class BModule(Module):
    """This class only exists to silence some pyright messages

    You can access buffers by `obj.b.buffername`. They will be known to be
    Tensors (not Modules), and pyright will stop painting the town red.

    Also, sometimes you want to have a tensor which does not get .to()d
    around. It's too big. It can stay on cpu with register_cpu_buffer().
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # bypassing torch Module's hooks
        self.___cpu_buffers = {}
        self._init_bgetter()

    def _init_bgetter(self):
        self.___bgetter = BufGetter(self._buffers, self.___cpu_buffers)

    @property
    def b(self) -> "BufGetter":
        return self.___bgetter

    def register_cpu_buffer(self, name: str, buf: Tensor | None):
        self.___cpu_buffers[name] = buf

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

    def __getstate__(self):
        del self.___bgetter
        state = super().__getstate__()
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._init_bgetter()


class BufGetter:
    __slots__ = "buffers", "cpu_buffers"

    def __init__(self, buffers, cpu_buffers):
        self.buffers = buffers
        self.cpu_buffers = cpu_buffers

    def __getattr__(self, key) -> Tensor:
        if key in self.cpu_buffers:
            return self.cpu_buffers[key]
        elif key in self.buffers:
            return self.buffers[key]
        raise AttributeError(f"BufGetter didn't find {key=}.")

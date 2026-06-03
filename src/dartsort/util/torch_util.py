import gc
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import Module

from .logging_util import DARTSORTVERBOSE, get_logger

if TYPE_CHECKING:
    from .internal_config import ComputationConfig

logger = get_logger(__name__)


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
        self.___cpu_buffers: dict[str, Tensor | None] = {}
        self._init_bgetter()

    def del_none_buffer(self, bufname: str):
        assert bufname in self.___cpu_buffers
        assert hasattr(self, bufname)
        assert self.___cpu_buffers[bufname] is None
        delattr(self, bufname)
        del self.___cpu_buffers[bufname]

    def _init_bgetter(self):
        self.___bgetter = BufGetter(self._buffers, self.___cpu_buffers)

    @property
    def b(self) -> "BufGetter":
        return self.___bgetter

    def get_optional_buffer(self, name: str) -> Tensor | None:
        return getattr(self.b, name)

    def register_cpu_buffer(self, name: str, buf: Tensor | None):
        self.___cpu_buffers[name] = buf

    def register_buffer_or_none(
        self,
        name: str,
        buf: Tensor | None,
        on_device: bool = True,
        persistent=True,
    ):
        if buf is None:
            setattr(self, name, None)
            self.register_cpu_buffer(name, None)
        elif on_device:
            self.register_buffer(name, buf, persistent=persistent)
        else:
            self.register_cpu_buffer(name, buf)

    def __getstate__(self):
        try:
            del self.___bgetter
        except AttributeError:
            pass
        state = super().__getstate__()
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._init_bgetter()


class BufGetter:
    __slots__ = "buffers", "cpu_buffers"

    def __init__(self, buffers, cpu_buffers):
        super().__setattr__("buffers", buffers)
        super().__setattr__("cpu_buffers", cpu_buffers)

    def __setattr__(self, key, value):
        # don't set my properties directly, but += etc are allowed and will
        # modify the property in-place, so that ids are equal
        if key in self.cpu_buffers:
            assert value is self.cpu_buffers[key]
        elif key in self.buffers:
            assert value is self.buffers[key]
        else:
            raise AttributeError

    def __getattr__(self, key) -> Tensor:
        if key in self.cpu_buffers:
            return self.cpu_buffers[key]
        elif key in self.buffers:
            return self.buffers[key]
        raise AttributeError(f"BufGetter didn't find {key=}.")


def cleanup_and_log_gpu_usage(computation_cfg: "ComputationConfig", message=""):
    dev = computation_cfg.actual_device()
    gc.collect()

    if dev.type != "cuda":
        return

    torch.cuda.empty_cache()

    if logger.isEnabledFor(DARTSORTVERBOSE):
        message = (
            f"{message}\n{torch.cuda.memory_summary(device=dev, abbreviated=True)}"
        )
        logger.dartsortverbose(message)


_logged_compile_thing = False


def torch_compile(fn, dynamic=True, fullgraph=True):
    """Try to use torch.compile if GPU is supported, else fall back on jit.script.

    But they don't like us to use script anymore and say it's not supported on
    Python 3.14 and later. Still, I have an old GPU personally so it helps me to
    do this.

    Here I'm using dynamic=True and fullgraph=True to try to be similar to how
    script works, because I like the strictness but also this is used for little
    utility functions where the input shapes change all the time.
    """
    # clearly I don't understand what compile does.
    return torch.jit.script(fn)

    # global _logged_compile_thing
    # if not torch.cuda.is_available():
    #     return torch.compile(fn, dynamic=dynamic, fullgraph=fullgraph)
    # major, _ = torch.cuda.get_device_capability()
    # if major >= 7:
    #     return torch.compile(fn, dynamic=dynamic, fullgraph=fullgraph)
    # else:
    #     if not _logged_compile_thing:
    #         logger.dartsortdebug(
    #             "Falling back on torch.jit.script since GPU's capability version "
    #             f"is {major}. You might see a deprecation warning from torch "
    #             "depending on your Python version."
    #         )
    #         _logged_compile_thing = True
    #     return torch.jit.script(fn)


def torch_compiler(dynamic=True, fullgraph=True):
    def dec(fn):
        return torch_compile(fn, dynamic=dynamic, fullgraph=fullgraph)
    return dec

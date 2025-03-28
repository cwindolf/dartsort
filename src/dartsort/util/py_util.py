import signal
import time
from importlib.metadata import Distribution
import sys
import json
from pathlib import Path


# check if we are installed in editable mode
pkgname = sys.modules[__name__].__name__.split(".")[0]
durl = Distribution.from_name(pkgname).read_text("direct_url.json")
EDITABLE = json.loads(durl).get("dir_info", {}).get("editable", False)


class timer:
    """
    with timer("hi"):
        bubblesort(np.arange(1e6)[::-1])
    # prints: hi took <> s
    with timer("zoom") as tic:
        pass
    assert np.isclose(tic.dt, 0)
    """

    def __init__(self, name="timer"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.dt = time.time() - self.start
        print(self.name, "took", self.dt, "s")


class NoKeyboardInterrupt:
    """A context manager that we use to avoid ending up in invalid states."""

    def handler(self, *sig):
        if self.sig:
            signal.signal(signal.SIGINT, self.old_handler)
            sig, self.sig = self.sig, None
            self.old_handler(*sig)
        self.sig = sig

    def __enter__(self):
        self.old_handler = signal.signal(signal.SIGINT, self.handler)
        self.sig = None

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.sig:
            self.old_handler(*self.sig)


delay_keyboard_interrupt = NoKeyboardInterrupt()


def int_or_inf(s):
    s = float(s)
    if np.isfinite(s):
        return int(s)
    return s


def resolve_path(p: str | Path, strict=False) -> Path:
    p = Path(p)
    p = p.expanduser()
    p = p.absolute()
    p = p.resolve(strict=strict)
    return p

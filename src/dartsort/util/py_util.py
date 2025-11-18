import contextlib
import dataclasses
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from importlib.metadata import Distribution
from pathlib import Path
from typing import dataclass_transform

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

# check if we are installed in editable mode
pkgname = sys.modules[__name__].__name__.split(".")[0]
durl = Distribution.from_name(pkgname).read_text("direct_url.json")
assert durl is not None
EDITABLE = json.loads(durl).get("dir_info", {}).get("editable", False)


# config.py and internal_config.py use the following decorator

pydantic_strict_cfg = ConfigDict(strict=True, extra="forbid")


# needed to annotate pydantic for pyright to pick up cfg fields
@dataclass_transform(kw_only_default=True, frozen_default=True)
def cfg_dataclass(*args, frozen=True, kw_only=True, **kwargs):
    return dataclass(
        *args, **kwargs, frozen=frozen, kw_only=kw_only, config=pydantic_strict_cfg
    )


# lightweight dataclass defaults
@dataclass_transform(kw_only_default=True, eq_default=False)
def databag(*args, slots=True, kw_only=True, eq=False, repr=False, **kwargs):
    return dataclasses.dataclass(
        *args, **kwargs, slots=slots, kw_only=kw_only, eq=eq, repr=repr
    )


# random utility classes


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
            self.old_handler(*sig)  # type: ignore
        self.sig = sig

    def __enter__(self):
        # TODO: maybe should just handle this in the peeling code. would need to detect
        # partially saved batches somehow... uhh... maybe based on the last sample saved,
        # if we update that last and resume from there...
        if threading.current_thread() is threading.main_thread() and os.name == "posix":
            self.old_handler = signal.signal(signal.SIGINT, self.handler)
            self.sig = None

    def __exit__(self, type, value, traceback):
        if threading.current_thread() is threading.main_thread() and os.name == "posix":
            signal.signal(signal.SIGINT, self.old_handler)
            if self.sig:
                self.old_handler(*self.sig)  # type: ignore


if threading.current_thread() is threading.main_thread() and os.name == "posix":
    delay_keyboard_interrupt = NoKeyboardInterrupt()
else:
    delay_keyboard_interrupt = contextlib.nullcontext()


# helper functions used as `type`s in argparse


def int_or_inf(s):
    s = float(s)
    if math.isfinite(s):
        return int(s)
    return s


def int_or_float(s):
    s = s.strip()
    if not s.strip("0123456789"):
        return int(s)
    return float(s)


def int_or_float_or_none(s):
    s = s.strip()
    if s.lower() in ("none", ""):
        return None
    if not s.strip("0123456789"):
        return int(s)
    return float(s)


def float_or_none(s):
    s = s.strip()
    if s.lower() in ("none", ""):
        return None
    return float(s)


def int_or_none(s):
    s = s.strip()
    if s.lower() in ("none", ""):
        return None
    return int(s)


def float_or_str(s):
    try:
        return float(s)
    except ValueError:
        return s


def str_or_none(s):
    if s.lower() == "none":
        return None
    return s


# files and paths


def resolve_path(p: str | Path | None, strict=False) -> Path:
    if p is None:
        raise ValueError("Can't resolve path None.")
    p = Path(p)
    p = p.expanduser()
    p = p.absolute()
    p = p.resolve(strict=strict)
    return p


def dartcopy2(icfg, src, dest):
    if icfg.workdir_copier == "shutil":
        try:
            shutil.copy2(src, dest, follow_symlinks=icfg.workdir_follow_symlinks)
        except shutil.SameFileError:
            # this happens in a symlink workflow that I use sometimes
            return
    elif icfg.workdir_copier == "rsync":
        _rsync(src, dest, archive=False, follow_symlinks=icfg.workdir_follow_symlinks)
    else:
        assert False


def dartcopytree(icfg, src, dest):
    if icfg.workdir_copier == "shutil":
        shutil.copytree(
            src,
            dest,
            symlinks=not icfg.workdir_follow_symlinks,
            dirs_exist_ok=True,
        )
    elif icfg.workdir_copier == "rsync":
        _rsync(
            f"{src}/",
            f"{dest}/",
            archive=True,
            follow_symlinks=icfg.workdir_follow_symlinks,
        )
    else:
        assert False


def _rsync(src, dest, archive=True, follow_symlinks=False):
    archive_flags = ["-a"] if archive else []
    link_flags = ["--no-links", "-L"] if follow_symlinks else []
    res = subprocess.run(["rsync", *archive_flags, *link_flags, str(src), str(dest)])
    assert not res.returncode

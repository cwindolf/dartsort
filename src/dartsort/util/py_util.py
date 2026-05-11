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
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import dataclass_transform

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from .logging_util import DARTSORTVERBOSE, get_logger

logger = get_logger(__name__)


# without this, pydantic allows unknown keys, so you can easily
# make a typo in your parameter name!
pydantic_strict_cfg = ConfigDict(strict=True, extra="forbid")


# used for configuration objects in [internal_]config.py
@dataclass_transform(kw_only_default=True, frozen_default=True)
def cfg_dataclass(*args, frozen=True, kw_only=True, **kwargs):
    return pydantic_dataclass(
        *args, **kwargs, frozen=frozen, kw_only=kw_only, config=pydantic_strict_cfg
    )


# lightweight dataclass defaults
@dataclass_transform(kw_only_default=True, eq_default=False)
def databag(*args, slots=True, kw_only=True, eq=False, repr=False, **kwargs):
    return dataclasses.dataclass(
        *args, **kwargs, slots=slots, kw_only=kw_only, eq=eq, repr=repr
    )


# random utility classes

_timer_stack = []


class timer:
    """
    with timer("hi"):
        bubblesort(np.arange(1e6)[::-1])
    # prints: hi took rot90(8) s
    with timer("zoom", {}) as tic:
        with timer("zip") as tac:
            pass
    assert np.isclose(tac.dt, 0)
    tic.results_dict # => nested timings
    """

    def __init__(self, name="timer", results_dict=None, loglevel=DARTSORTVERBOSE):
        self.loglevel = loglevel
        self.name = name
        self.results_dict = results_dict
        self.parent = None

    def start(self):
        self.t0 = time.time()

    def stop(self):
        self.dt = time.time() - self.t0
        logger.log(self.loglevel, "%s took %ss", self.name, self.dt)
        if self.parent is not None and self.results_dict is not None:
            self.results_dict[f"{self.parent.name}: {self.name}"] = self.dt
        elif self.results_dict is not None:
            self.results_dict[self.name] = self.dt

    def __enter__(self):
        global _timer_stack
        if len(_timer_stack):
            self.parent = _timer_stack[-1]
            if self.results_dict is None:
                self.results_dict = self.parent.results_dict
        _timer_stack.append(self)
        self.start()
        return self

    def __exit__(self, *args):
        global _timer_stack
        self.stop()
        assert _timer_stack.pop() is self
        self.parent = None


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


def resolve_path(
    p: str | Path | Traversable | None, strict=False, mkdir=False, parents=False
) -> Path:
    if p is None:
        raise ValueError("Can't resolve path None.")
    if isinstance(p, Traversable):
        assert isinstance(p, Path)
    p = Path(p)
    p = p.expanduser()
    p = p.absolute()
    p = p.resolve(strict=strict)
    if mkdir:
        p.mkdir(parents=parents, exist_ok=True)
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
    try:
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
    except shutil.SameFileError:
        logger.dartsortdebug(
            f"Skip dartcopytree {src} -> {dest} since shutil says they're the same."
        )
    except shutil.Error as e:
        # Sometimes the same file error is hiding in this.
        # This is probably not the right way to do this?
        # It's a weird exception format though...
        arg = e.args[0]
        if not all(isinstance(a, str) and len(a) == 1 for a in arg):
            raise
        arg = "".join(arg)
        if not arg.startswith("<DirEntry"):
            raise
        if not arg.endswith("are the same file"):
            raise
        logger.dartsortdebug(
            f"shutil.copytree said (re: {src} and {dest}) that {arg}. "
            "Ignoring that and continuing."
        )
    except Exception as e:
        raise ValueError(
            f"dartcopytree {src} -> {dest} failed. {src.exists()=}, {dest.exists()=}."
        ) from e


def _rsync(src, dest, archive=True, follow_symlinks=False):
    archive_flags = ["-a"] if archive else []
    link_flags = ["--no-links", "-L"] if follow_symlinks else []
    res = subprocess.run(["rsync", *archive_flags, *link_flags, str(src), str(dest)])
    assert not res.returncode

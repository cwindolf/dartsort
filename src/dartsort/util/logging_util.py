import os
import warnings
from logging import (
    DEBUG,
    INFO,
    NOTSET,
    addLevelName,
    getLevelNamesMapping,
    getLogger,
    getLoggerClass,
    setLoggerClass,
)
from time import perf_counter

from tqdm.auto import tqdm as auto_tqdm
from tqdm.auto import trange as auto_trange
from tqdm.notebook import tqdm as notebook_tqdm

DARTSORTVERBOSE = DEBUG + 4
addLevelName(DARTSORTVERBOSE, "DSVERBOSE")

DARTSORTDEBUG = DEBUG + 5
addLevelName(DARTSORTDEBUG, "DSDEBUG")


DOUBLECHECK = DEBUG + 6
addLevelName(DOUBLECHECK, "DOUBLECHECK")


class DARTsortLogger(getLoggerClass()):
    def __init__(self, name, level=NOTSET):
        super().__init__(name, level)

    def doublecheck(self, msg, *args, **kwargs):
        if self.isEnabledFor(DOUBLECHECK):
            self._log(DOUBLECHECK, msg, args, stacklevel=2, **kwargs)

    def dartsortverbose(self, msg, *args, **kwargs):
        if self.isEnabledFor(DARTSORTVERBOSE):
            self._log(DARTSORTVERBOSE, msg, args, stacklevel=2, **kwargs)

    def dartsortdebug(self, msg, *args, **kwargs):
        if self.isEnabledFor(DARTSORTDEBUG):
            self._log(DARTSORTDEBUG, msg, args, stacklevel=2, **kwargs)

    def dartsortdebugthunk(self, msg, *args, **kwargs):
        if self.isEnabledFor(DARTSORTDEBUG):
            self._log(DARTSORTDEBUG, msg(), args, stacklevel=2, **kwargs)


setLoggerClass(DARTsortLogger)


# shouts out to sinclairtarget.com
package_logger = getLogger(__package__)
assert isinstance(package_logger, DARTsortLogger)


# set to environment-defined log level if present
if (level := os.getenv("LOGLEVEL")) is not None:
    pass
elif (level := os.getenv("LOG_LEVEL")) is not None:
    pass

if level:
    level = level.strip()
    if not level.strip("0123456789"):
        ilevel = int(level)
    else:
        ilevel = getLevelNamesMapping()[level.upper()]
    package_logger.setLevel(ilevel)
    package_logger.log(ilevel, f"Log level set to {level} ({ilevel}).")


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    import sys
    import traceback

    log = file if hasattr(file, "write") else sys.stderr
    assert log is not None
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


# override warnings to show tracebacks when debugging
if package_logger.isEnabledFor(DARTSORTVERBOSE):
    package_logger.dartsortdebug("Setting warnings.showwarning to print tracebacks.")
    warnings.showwarning = warn_with_traceback  # type: ignore


def get_logger(*args, **kwargs) -> DARTsortLogger:
    logger = getLogger(*args, **kwargs)
    assert isinstance(logger, DARTsortLogger)
    return logger


package_logger.dartsortdebug(
    f"Logger is enabled for: DARTSORTDEBUG={package_logger.isEnabledFor(DARTSORTDEBUG)}, "
    f"DARTSORTVERBOSE={package_logger.isEnabledFor(DARTSORTVERBOSE)}."
)


class logress:
    def __init__(
        self,
        iterable,
        logger=package_logger,
        miniters=100,
        mininterval=60.0,
        desc=None,
        total=None,
        smoothing=0.0,
        unit="it",
        level=INFO,
        initial=0,
        miniters_fraction=0.2,
    ):
        del smoothing

        self.iterable = iterable
        self.desc = desc
        self.level = level
        self.miniters = max(miniters, 1)
        self.mininterval = mininterval
        self.logger = logger
        self.unit = unit
        self.closed = False
        try:
            self.total = len(iterable)
            self.miniters = min(
                self.miniters, max(1, int(miniters_fraction * self.total))
            )
        except TypeError:
            self.total = total

        self.n = initial
        self.start = perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __iter__(self):
        it = self.iterable
        if not self.logger.isEnabledFor(self.level):
            yield from it
            return

        n = self.n
        miniters = self.miniters
        mininterval = self.mininterval

        last_n = n
        start = perf_counter()
        tic = start
        nstart = n + 1

        self.last_n = n
        self.start = start
        self.tic = tic
        try:
            for o in it:
                yield o
                n += 1
                if n > nstart and n - last_n < miniters:
                    continue
                toc = perf_counter()
                if n > nstart and (toc - tic) < mininterval:
                    continue
                self.n = last_n = n
                self._print(t=toc, check=False)
                tic = toc
        finally:
            self.close()
            self.n = self.last_n = n
            self.tic = tic

    def set_description(self, desc, refresh=False):
        self.desc = desc
        if refresh:
            self._print()

    def update(self, n):
        self.n = n
        self._print()

    def write(self, s):
        self.logger.log(self.level, s)

    def close(self):
        if self.closed:
            return
        self._print(check=False)
        self.closed = True

    def _print(self, t=None, check=True):
        if t is None:
            t = perf_counter()

        n = self.n
        if check:
            if n - self.last_n < self.miniters:
                return
            if (t - self.tic) < self.mininterval:
                return
        elif n == self.last_n:
            return

        dt = t - self.start
        rate = n / dt
        if rate >= 1.0:
            rate_str = f"{rate:.1f}{self.unit}/s"
        else:
            rate_str = f"{dt / n:.1f}s/{self.unit}"
        elapsed = seconds_str(dt)
        if self.total:
            prop = n / self.total
            pct = 100 * prop
            prog_str = f"{n}/{self.total} ({pct:.1f}%)"

            eta = (1 - prop) * (dt / prop)
            eta = seconds_str(eta)
            time_str = f"{elapsed}<{eta}"
        else:
            prog_str = f"{n}/?"
            time_str = f"{elapsed}/?"

        self.write(f"{self.desc}: [{prog_str}] ({rate_str}, {time_str})")

        self.last_n = n
        self.tic = t


_is_notebook = issubclass(auto_tqdm, notebook_tqdm)
if _is_notebook:
    progbar = auto_tqdm
    progrange = auto_trange
else:
    progbar = logress

    def progrange(*args, **kwargs):
        return logress(range(*args), **kwargs)


def seconds_str(dt):
    hours, remainder = divmod(dt, 3600)
    hours = int(hours)
    minutes, seconds = divmod(remainder, 60)
    minutes = int(minutes)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:04.1f}"
    elif minutes:
        return f"{minutes:02d}:{seconds:04.1f}"
    else:
        return f"{seconds:0.2f}s"

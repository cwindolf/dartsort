import os
from logging import (
    getLoggerClass,
    addLevelName,
    setLoggerClass,
    NOTSET,
    DEBUG,
    getLogger,
    getLevelNamesMapping,
    basicConfig,
)
import warnings


DARTSORTVERBOSE = DEBUG + 4
addLevelName(DARTSORTVERBOSE, "DARTSORTVERBOSE")

DARTSORTDEBUG = DEBUG + 5
addLevelName(DARTSORTDEBUG, "DARTSORTDEBUG")


DOUBLECHECK = DEBUG + 6
addLevelName(DOUBLECHECK, "DOUBLECHECK")


class DARTsortLogger(getLoggerClass()):
    def __init__(self, name, level=NOTSET):
        super().__init__(name, level)

    def doublecheck(self, msg, *args, **kwargs):
        if self.isEnabledFor(DOUBLECHECK):
            self._log(DOUBLECHECK, msg, args, **kwargs)

    def dartsortverbose(self, msg, *args, **kwargs):
        if self.isEnabledFor(DARTSORTVERBOSE):
            self._log(DARTSORTVERBOSE, msg, args, **kwargs)

    def dartsortdebug(self, msg, *args, **kwargs):
        if self.isEnabledFor(DARTSORTDEBUG):
            self._log(DARTSORTDEBUG, msg, args, **kwargs)

    def dartsortdebugthunk(self, msg, *args, **kwargs):
        if self.isEnabledFor(DARTSORTDEBUG):
            self._log(DARTSORTDEBUG, msg(), args, **kwargs)


setLoggerClass(DARTsortLogger)


logger: DARTsortLogger = getLogger(__name__)


# set to environment-defined log level if present
if "LOG_LEVEL" in os.environ:
    level = os.environ["LOG_LEVEL"]
    try:
        basicConfig(level=level)
    except ValueError:
        ilevel = int(level)
        basicConfig(level=ilevel)
    else:
        ilevel = getLevelNamesMapping()[level]
    logger.log(ilevel, f"Log level set to {level} ({ilevel}).")


# override warnings to show tracebacks when debugging
if logger.isEnabledFor(DARTSORTDEBUG):
    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        import sys, traceback

        log = file if hasattr(file, "write") else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    logger.dartsortdebug("Setting warnings.showwarning to print tracebacks.")
    warnings.showwarning = warn_with_traceback

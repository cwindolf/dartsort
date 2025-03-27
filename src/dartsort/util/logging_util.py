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


if "LOG_LEVEL" in os.environ:
    level = os.environ["LOG_LEVEL"]
    try:
        basicConfig(level=level)
    except ValueError:
        ilevel = int(level)
        basicConfig(level=ilevel)
    else:
        ilevel = getLevelNamesMapping()[level]
    logger = getLogger(__name__)
    logger.log(ilevel, f"Log level set to {level} ({ilevel}).")

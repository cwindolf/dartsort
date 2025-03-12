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

DARTSORTDEBUG = DEBUG + 5
addLevelName(DARTSORTDEBUG, "DARTSORTDEBUG")


class DARTsortLogger(getLoggerClass()):
    def __init__(self, name, level=NOTSET):
        super().__init__(name, level)

    def dartsortdebug(self, msg, *args, **kwargs):
        if self.isEnabledFor(DARTSORTDEBUG):
            self._log(DARTSORTDEBUG, msg, args, **kwargs)


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
    logger = getLogger()
    logger.log(ilevel, f"Log level set to {level} ({ilevel}).")

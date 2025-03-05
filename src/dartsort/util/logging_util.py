from logging import getLoggerClass, addLevelName, setLoggerClass, NOTSET, DEBUG

DARTSORTDEBUG = DEBUG + 5
addLevelName(DARTSORTDEBUG, "DARTSORTDEBUG")


class DARTsortLogger(getLoggerClass()):
    def __init__(self, name, level=NOTSET):
        super().__init__(name, level)

    def dartsortdebug(self, msg, *args, **kwargs):
        if self.isEnabledFor(DARTSORTDEBUG):
            self._log(DARTSORTDEBUG, msg, args, **kwargs)


setLoggerClass(DARTsortLogger)
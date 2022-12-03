"""Create globally-acessible logger object."""

import logging
import os
import sys

import pandas as pd

from logging import _srcfile
from pathlib import Path


class DFLogger(logging.Logger):
    """See base class (this handles pandas objects better)."""
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False,
             stacklevel=1):
        """
        Override this method to better handle logging of DataFrames.
        """
        if type(msg) in (pd.DataFrame, pd.Series):
            # Nicer to record dataframes properly.
            msg = '--- DataFrame with contents ---\n\t{}'.format(
                msg.to_string().replace('\n', '\n\t'))
        sinfo = None
        if _srcfile:
            try:
                fn, lno, func, sinfo = self.findCaller(stack_info, stacklevel)
            except ValueError: # pragma: no cover
                fn, lno, func = "(unknown file)", 0, "(unknown function)"
        else: # pragma: no cover
            fn, lno, func = "(unknown file)", 0, "(unknown function)"
        if exc_info:
            if isinstance(exc_info, BaseException):
                exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
            elif not isinstance(exc_info, tuple):
                exc_info = sys.exc_info()
        record = self.makeRecord(self.name, level, fn, lno, msg, args,
                                 exc_info, func, extra, sinfo)
        self.handle(record)

logging.setLoggerClass(DFLogger)

def get_logger(log_name, log_path=None, level=None):
    """Create a logging object available globally via the log_name."""

    _log_format = logging.Formatter(
        '{asctime} [{levelname}] [{module}:{lineno}] {name}: {message}',
        '%Y:%m:%d %H:%M:%S', style='{')

    logger = logging.getLogger(log_name)
    logger.propagate = False
    level = level or os.environ.get('LOGLEVEL', 'INFO').upper()
    logger.setLevel(level)

    if logging.StreamHandler not in [type(t) for t in logger.handlers]:
        h = logging.StreamHandler()
        h.setFormatter(_log_format)
        h.setLevel(level)
        logger.addHandler(h)

    if log_path is not None:
        f = logging.FileHandler(
            Path(log_path, 'output.log'), mode='w', encoding='utf-8')
        f.setLevel(level)
        f.setFormatter(_log_format)
        logger.addHandler(f)

    return logger

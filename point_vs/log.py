"""Create globally-acessible logger object."""

import logging


def create_log_obj(log_name, level=logging.INFO):
    """Create a logging object available globally via the log_name."""
    formatter = logging.Formatter(
        '{asctime}::{levelname}::{module}: {message}',
        '%Y:%m:%d %H:%M:%S',
        style='{')
    h = logging.StreamHandler()
    h.setFormatter(formatter)

    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    logger.addHandler(h)
    return logger

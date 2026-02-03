from __future__ import annotations

import logging
import sys


def test_get_logger_defined():
    from gwexpy.utils import logger as logger_mod

    assert hasattr(logger_mod, "get_logger")


def test_get_logger_configures_once():
    from gwexpy.utils.logger import get_logger

    name = "gwexpy.test.logger"
    raw = logging.getLogger(name)
    for h in list(raw.handlers):
        raw.removeHandler(h)

    logger = get_logger(name)
    assert isinstance(logger, logging.Logger)
    assert logger.name == name
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1

    handler = logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stdout
    assert isinstance(handler.formatter, logging.Formatter)

    logger2 = get_logger(name)
    assert logger2 is logger
    assert len(logger.handlers) == 1

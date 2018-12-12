_author__ = 'max'

import logging
import sys

class LogerCaches:
    loggers = {}

def get_logger(name, level=logging.INFO, handler=sys.stderr,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    if name not in LogerCaches.loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(formatter)
        stream_handler = logging.StreamHandler(handler)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        LogerCaches.loggers[name] = logger
    return LogerCaches.loggers[name]

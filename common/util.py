import os
import sys
import time
import logging

from contextlib import contextmanager

from common.argparser import abstract


def get_logger(name, folder=os.path.join('logs', abstract)):
    """
    get a logger with std output and file output
    :param folder: logger folder
    :param name: logger name
    :return: logger
    """

    if not os.path.exists(folder):
        os.mkdir(folder)

    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)

    if os.path.exists(os.path.join(folder, '{}.log'.format(name))):
        os.remove(os.path.join(folder, '{}.log'.format(name)))

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s\tmodule:{}\t%(message)s'.format(name))

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(os.path.join(folder, '{}.log'.format(name)))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


@contextmanager
def timed(msg, logger):
    tstart = time.time()
    yield
    logger.info('%s done in %.3f seconds' % (msg, time.time() - tstart))

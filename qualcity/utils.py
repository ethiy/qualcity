#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import operator
import functools

import logging

import sys

import unicodedata

import numpy as np

utils_logger = logging.getLogger(__name__)


def normalize_caseless(string):
    return unicodedata.normalize("NFKD", string.casefold())


def caseless_equal(lhs, rhs):
    return normalize_caseless(lhs) == normalize_caseless(rhs)


def check_iterable(iterable):
    utils_logger.debug('Checking if %s is iterable.', iterable)
    if not hasattr(iterable, '__iter__'):
        utils_logger.critical('Rasing AttributeError error...')
        raise AttributeError


def median(iterable):
    utils_logger.debug('Computing the median of %s', iterable)
    check_iterable(iterable)
    length = len(iterable)
    sorted_iterable = sorted(iterable)
    if length == 0:
        raise LookupError
    elif length % 2 == 1:
        return sorted_iterable.__getitem__(length // 2)
    else:
        return (
            sorted_iterable.__getitem__(length // 2 - 1) +
            sorted_iterable.__getitem__(length // 2)
        ) / 2.


def mean(iterable):
    utils_logger.debug('Computing the mean of %s', iterable)
    length = len(iterable)

    if length == 0:
        raise LookupError
    else:
        return sum(iterable) / len(iterable)


def histogram(iterable, **kwargs):
    utils_logger.debug('Computing histogram for %s', iterable)
    check_iterable(iterable)
    return np.histogram(np.array(iterable), **kwargs)


def stat(statistic):
    utils_logger.debug(
        'Getting statistic function correponding to %s...',
        statistic
    )
    try:
        return {
            'min': min,
            'max': max,
            'mean': mean,
            'median': median
        }[statistic]
    except KeyError:
        utils_logger.exception(
            '%s is not recognized as a statistic here:',
            statistic
        )


def stats(attribute, statistics, **kwargs):
    utils_logger.debug('Getting %s of %s...', statistics, attribute)
    if statistics == 'histogram':
        utils_logger.info('Getting histogram...')
        return list(histogram(attribute, **kwargs)[0])
    elif type(statistics) is list:
        utils_logger.info('Getting the statistics list %s...', statistics)
        return functools.reduce(
            lambda _list, stat: (
                _list
                +
                [
                    stat(attribute)
                ]
            ),
            [stat(statistic) for statistic in statistics if callable(stat)],
            []
        )
    else:
        raise LookupError


def fuse(dict_1, dict_2):
    utils_logger.debug('Fusing %s and %s', dict_1, dict_2)
    k1, k2 = map(
        lambda d: d.keys(),
        [dict_1, dict_2]
    )
    return dict(
        [
            (key, (dict_1[key], dict_2[key]))
            for key in set(k1) & set(k2)
        ]
        +
        [
            (key, (dict_1[key], None))
            for key in set(k1) - set(k2)
        ]
        +
        [
            (key, (None, dict_2[key]))
            for key in set(k2) - set(k1)
        ]
    )


def resolve(string):
    """
        resolve(string)
    """
    module, *names = string.split('.')
    try:
        imported = __import__(module)
        for name in names:
            module += '.' + name
            try:
                imported = getattr(imported, name)
            except AttributeError:
                __import__(module)
                imported = getattr(imported, name)
        return imported
    except ImportError:
        _, exc, tb = sys.exc_info()
        v_exc = ValueError('Could not resolve %r: %s' % (string, exc))
        v_exc.__cause__, v_exc.traceback__ = exc, tb
        raise v_exc

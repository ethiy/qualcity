#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import operator
import functools

import logging

import sys

import numpy as np

utils_logger = logging.getLogger('qualcity.' + __name__)


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


def main():
    print(median([-5, -5, -3, -4, 0, -1]))
    print(median((-5, -5, -3, -4, 0, -1)))
    print(mean((-5, -5, -3, -4, 0, -1)))
    print(median([-5, -5, -4, 0, -1]))
    print(mean([-5, -5, -4, 0, -1]))

    print(
        fuse(
            {'4': 545, '10': 641, '8': 45},
            {'4': 15, '10': 8574, '7': 9473}
        )
    )

    stat(max)
    print(
        stats([12, 5, 0, 11.1], ['max', 'min', 'histogram'])
    )
    print(
        stats([12, 5, 0, 11.1], 'histogram', bins=range(0, 20))
    )

    print(resolve('sklearn.decomposition.PCA').__doc__)
    resolve('sklearn.decomp.PCA')


if __name__ == '__main__':
    main()

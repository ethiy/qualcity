#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import operator
import functools

import logging

import numpy as np


def check_iterable(iterable):
    logging.debug('Checking if %s is iterable.', iterable)
    if not hasattr(iterable, '__iter__'):
        logging.critical('Rasing AttributeError error...')
        raise AttributeError


def median(iterable):
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
    return sum(iterable) / len(iterable)


def histogram(iterable, **kwargs):
    logging.debug('Computing histogram for %s', iterable)
    check_iterable(iterable)
    return np.histogram(np.array(iterable), **kwargs)


def stat(statistic):
    logging.debug(
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
        logging.error('%s is not recognized as a statistic here', statistic)


def stats(attribute, statistics, **kwargs):
    logging.debug('Getting %s of %s...', statistics, attribute)
    if statistics == 'histogram':
        logging.debug('Getting histogram...')
        return histogram(attribute, **kwargs)
    elif type(statistics) is list:
        return functools.reduce(
            lambda _list, stat: (
                _list
                +
                [
                    stat(attribute)
                ]
            ),
            filter(
                lambda stat: callable(stat),
                [stat(statistic) for statistic in statistics]
            ),
            []
        )
    else:
        logging.error('%s not recognized', statistics)


def fuse_elements(elements):
    logging.debug('Sorting dictionnaries based on their length.')
    s, elements = zip(
        *sorted(
            enumerate(elements),
            key=lambda c: len(c[1])
        )
    )

    logging.debug('Separating keys and values.')
    keys, values = zip(
        *map(
            lambda el: zip(*el),
            elements
        )
    )

    if not set(keys[0]) <= set(keys[1]):
        logging.critical('The dictionnaries cannot be fused by keys!')
        raise LookupError

    logging.debug(
        'Getting the values in the bigest dictionnary corresponding to the '
        +
        'keys that are present in the smallest...'
    )
    values_2 = [val for key, val in elements[1] if key in set(keys[0])]

    return list(
            zip(
                keys[0],
                [
                    (
                        couple[s[0]],
                        couple[s[1]]
                    )
                    for couple
                    in zip(
                        values[0],
                        values_2
                    )
                ]
            )
        ) + [
            (key, (None, val))
            if s[0] == 0
            else (key, (val, None))
            for key, val in elements[1]
            if key not in set(keys[0])
        ]


def fuse(dict_1, dict_2):
    logging.debug('Fusing %s and %s', dict_1, dict_2)
    logging.debug('Sorting each dictionnary based its keys.')
    return dict(
        fuse_elements(
            map(
                lambda d: sorted(d, key=operator.itemgetter(0)),
                [dict_1.items(), dict_2.items()]
            )
        )
    )


def main():
    print(median([-5, -5, -3, -4, 0, -1]))
    print(median((-5, -5, -3, -4, 0, -1)))
    print(mean((-5, -5, -3, -4, 0, -1)))
    print(median([-5, -5, -4, 0, -1]))
    print(mean([-5, -5, -4, 0, -1]))

    print(
        fuse(
            {'4': 545, '10': 641, '8': 45},
            {'4': 15, '10': 8574}
        )
    )

    stat(max)
    print(
        stats([12, 5, 0, 11.1], ['max', 'min', 'histogram'])
    )
    print(
        stats([12, 5, 0, 11.1], 'histogram', bins=range(0, 20))
    )


if __name__ == '__main__':
    main()

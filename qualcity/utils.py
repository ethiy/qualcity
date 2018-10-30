# -*- coding: <utf-8> -*-

import operator
import functools

import logging

import sys
import os
import ast
import uuid

import unicodedata

import numpy as np
import statistics

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


def chunk(iterable, size):
    check_iterable(iterable)
    return [iterable[i: i + size] for i in range(0, len(iterable), size)]


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
            'mean': statistics.mean,
            'median': statistics.median,
            'std': statistics.stdev
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
                ] if callable(stat) else None
            ),
            [stat(statistic) for statistic in statistics if callable(stat)],
            []
        )
    else:
        raise LookupError('%s unknown', statistics)


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


def cache_ledger(cache_dir, kind):
    try:
        with open(
                os.path.join(
                    cache_dir,
                    kind,
                    'codebook.csv'
                ),
                'r'
            ) as codebook:
            lines = codebook.readlines()
        
        return dict(
            [line.split('; ') for line in lines]
        )
    except FileNotFoundError:
        return {} 


def cached(config, cachebook):
    return [
        cache_name
        for (cache_name, cache_conf) in cachebook.items()
        if ast.literal_eval(cache_conf) == config
    ]


def store_cache_ledger(cache_dir, kind, cache_id, configuration):
    with open(
            os.path.join(
                cache_dir,
                kind,
                'codebook.csv'
            ),
            'a+'
        ) as codebook:
        codebook.write(
            cache_id
            +
            '; '
            +
            str(configuration)
            +
            '\n'
        )

    if kind not in ['classifiers', 'predictions', 'features']:
        utils_logger.warning('Unknown %s cached variable type!', kind)


def cache_features(cache_dir, feat_type, config, feat_dict):
    for (building, feature) in feat_dict.items():
        cache_feature(cache_dir, feat_type, building, config, feature)


def cache_feature(cache_dir, feat_type, building, config, feature):
    feature_id = str(uuid.uuid4())
    store_cache_ledger(
        cache_dir,
        'features',
        feature_id, 
        dict(
            [
                (
                    'type',
                    feat_type
                ),
                (
                    'building',
                    building
                )
            ]
            +
            list(config.items())
        )
    )
    np.savetxt(
        os.path.join(
            cache_dir,
            'features',
            feature_id + '.csv'
        ),
        np.expand_dims(feature, axis=0),
        delimiter=","
    )


def read_cached_feature(cache_dir, config, cachebook):
    cachedname_ = cached(config, cachebook)
    return read_feature(cache_dir, cachedname_[0]) if len(cachedname_) else None


def read_feature(cache_dir, cachename):
    with open(
        os.path.join(
            cache_dir,
            'features',
            cachename + '.csv'
        ),
        'r'
        ) as cache_file:
        line = cache_file.readline()
    return np.fromstring(line, sep=',')

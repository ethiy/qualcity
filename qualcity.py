#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

"""qualcity.

Usage:
    qualcity.py (-h | --help)
    qualcity.py pipeline <config_file> [--verbose]

Options:
    -h --help           Show this screen.

"""

import docopt

import time

import functools

import yaml

import logging

import altimetric_difference
import geometry_io
import labels_io
import utils

logger = logging.getLogger('qualcity')


def altimetric_features(depth, raster_dir, resolution):
    if depth < 1:
        raise ValueError
    return altimetric_difference.histogram_features(
        raster_dir,
        altimetric_difference.DSM_DIR,
        resolution,
        resolution
    )


def features(depth, feat_type, **kwargs):
    return {
        'geometric': lambda kwargs: geometry_io.geometric_features(**kwargs),
        'altimetric': lambda kwargs: altimetric_features(depth, **kwargs)
    }[feat_type](kwargs)


def get_features(depth, config):
    logger.info('Getting features ...')
    return functools.reduce(
        utils.fuse,
        [
            features(depth, feat_type, **config[feat_type])
            for feat_type in config.keys()
        ]
    )


def get_labels(hierarchical, depth, LoD, labels_dir):
    logger.info('Getting Labels ...')
    return labels_io.labels_map(
        labels_dir,
        hierarchical,
        depth,
        LoD,
        threshold=5
    )


def load_config(file):
    logger.info('Loading pipeline configuration file...')
    with open(file, mode='r') as conf:
        return yaml.load(conf)


def main():
    arguments = docopt.docopt(
        __doc__,
        help=True,
        version=None,
        options_first=False
    )

    logger.setLevel(logging.DEBUG if arguments['--verbose'] else logging.INFO)

    fh = logging.FileHandler('qualcity-' + time.ctime() + '.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    configuration = load_config(arguments['<config_file>'])
    logger.debug(yaml.dump(configuration))
    logger.info('Pipeline loaded.')

    labels = get_labels(
        **configuration['labels']
    )
    logger.debug('Labels are: %s', labels)
    logger.info('Labels safely loaded.')

    features = get_features(
        configuration['labels']['depth'],
        configuration['features']
    )
    logger.debug(features)
    logger.info('Features safely loaded.')


if __name__ == '__main__':
    main()

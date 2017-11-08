#! /usr/bin/env python2
# -*- coding: <utf-8> -*-

"""qualcity.

Usage:
    qualcity.py (-h | --help)
    qualcity.py pipeline <config_file> [--verbose]

Options:
    -h --help           Show this screen.

"""

from docopt import docopt

import yaml

import logging

import altimetric_difference
import geometry_io
import labels_io
import utils


def altimetric_features(level, raster_dir, granularity):
    if level < 1:
        raise ValueError
    return altimetric_difference.histogram_features(
        raster_dir,
        altimetric_difference.DSM_DIR,
        granularity,
        granularity
    )


def features(level, feat_type, **kwargs):
    return {
        'geometric': lambda kwargs: geometry_io.geometric_features(**kwargs),
        'altimetric': lambda kwargs: altimetric_features(level, **kwargs)
    }[feat_type](kwargs)


def get_features(level, config):
    logging.info('Getting features ...')
    return reduce(
        utils.fuse,
        [
            features(level, feat_type, **config[feat_type])
            for feat_type in config.keys()
        ]
    )


def get_labels(hierarchical, level, LoD, labels_dir):
    logging.info('Getting Labels ...')
    return labels_io.labels_map(
        labels_dir,
        hierarchical,
        level,
        LoD,
        threshold=5
    )


def load_config(file):
    logging.info('Loading pipeline configuration file...')
    with open(file, mode='r') as conf:
        return yaml.load(conf)


def main():
    arguments = docopt(__doc__, help=True, version=None, options_first=False)

    logging.basicConfig(
        level=logging.DEBUG if arguments['--verbose'] else logging.INFO
    )

    configuration = load_config(arguments['<config_file>'])
    logging.debug(yaml.dump(configuration))
    logging.info('Pipeline loaded.')

    labels = get_labels(
        **configuration['labels']
    )
    logging.debug('Labels are: %s', labels)
    logging.info('Labels safely loaded.')

    features = get_features(
        configuration['labels']['level'],
        configuration['features']
    )
    logging.debug(features)
    logging.info('Features safely loaded.')


if __name__ == '__main__':
    main()

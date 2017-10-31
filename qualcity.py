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

import altimetric_difference
import geometry_io
import utils


def altimetric_features(level, raster_dir, labels_dir, granularity):
    if level < 1:
        raise ValueError
    return altimetric_difference.histogram_features(
        raster_dir,
        labels_dir,
        altimetric_difference.DSM_DIR,
        granularity,
        granularity
    )


def features(level, feat_type, **kwargs):
    if level > 2:
        raise ValueError
    return {
        'geometric': geometry_io.geometric_features(**kwargs),
        'altimetric': altimetric_features(
            level,
            kwargs['raster_dir'],
            kwargs['labels_dir'],
            kwargs['granularity']
        )
    }[feat_type]


def get_features(level, config):
    return reduce(
        utils.fuse,
        [
            features(level, feat_type, **config[feat_type])
            for feat_type in config.keys()
        ]
    )


def get_labels(hierarchical, level, LoD, dir):
    return


def load_config(file):
    with open(file, mode='r') as conf:
        return yaml.load(conf)


def main():
    arguments = docopt(__doc__, help=True, version=None, options_first=False)

    configuration = load_config(arguments['<config_file>'])
    if arguments['--verbose']:
        print(yaml.dump(configuration))

    labels = get_labels(
        **configuration['labels']
    )

    feature_params = configuration['features']
    if 'altimetric' in feature_params:
        feature_params['altimetric']['labels'] = labels

    features = get_features(
        configuration['labels']['level'],
        feature_params
    )


if __name__ == '__main__':
    main()

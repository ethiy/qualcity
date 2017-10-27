#! /usr/bin/env python3
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
        'geometric': sorted(
            geometry_io.geometric_features(
                kwargs['graph_dir'],
                kwargs['attributes']
            ).iteritems(),
            key=operator.itemgetter(0)
        ),
        'altimetric': sorted(
            altimetric_features(
                level,
                kwargs['raster_dir'],
                kwargs['labels_dir'],
                kwargs['granularity']y
            ).iteritems(),
            key=operator.itemgetter(0)
        )
    }[feat_type]


def get_features(config):
    return reduce(
        utils.fuse,
        [
            features()
            for feat_type in
        ]
    )


def get_labels(hierarchical, level):
    return


def load_config(file):
    with open(file, mode='r') as conf:
        return yaml.load(conf)


def main():
    arguments = docopt(__doc__, help=True, version=None, options_first=False)
    if arguments['--verbose']:
        print(yaml.dump(load_config(arguments['<config_file>'])))


if __name__ == '__main__':
    main()

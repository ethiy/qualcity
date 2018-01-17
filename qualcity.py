#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

"""qualcity.

Usage:
    qualcity.py (-h | --help)
    qualcity.py pipeline <pipeline_config> [logger <logger_config>] [--verbose]

Options:
    -h --help           Show this screen.

"""

import docopt

import time

import functools
import operator

import yaml

import logging
import logging.config

import sklearn.decomposition
import sklearn.manifold
import sklearn.preprocessing

import sklearn.neural_network

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import altimetric_difference
import geometry_io
import labels_io
import utils

logger = logging.getLogger('qualcity')
default_config = {
    'version': 1,
    'formatters': {
        'verbose': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'simple': {
            'format': '%(levelname)s %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'WARN',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'formatter': 'verbose',
            'filename': 'qualcity-' + time.ctime() + '.log'
        }
    },
    'loggers': {
        'qualcity': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
        }
    }
}


def config_logger(arguments):
    if not arguments['logger']:
        logging.config.dictConfig(
            default_config
        )
        logger.info('Default logger chosen.')
    else:
        with open(arguments['<logger_config>'], mode='r') as conf:
            configuration = yaml.load(conf)

        configuration['handlers']['file']['filename'] = (
            'qualcity-' + time.ctime() + '.log'
        )
        configuration['loggers']['qualcity']['level'] = (
            'DEBUG' if arguments['--verbose'] else 'INFO'
        )
        logging.config.dictConfig(
            configuration
        )
        logger.info('Loaded logger from: ' + arguments['<logger_config>'])
        logger.debug(yaml.dump(configuration))


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


def format_features(depth, buildings, config):
    features = get_features(depth, config)
    features = [features[building] for building in buildings]

    if depth == 0:
        return [feature for feature, _ in features]
    else:
        return [np.hstack(feature) for feature in features]


def get_labels(hierarchical, depth, LoD, labels_dir):
    logger.info('Getting Labels ...')
    return [
        (building, label)
        for building, label in sorted(
            labels_io.labels_map(
                labels_dir,
                hierarchical,
                depth,
                LoD,
                threshold=5
            ).items(),
            key=operator.itemgetter(0)
        )
    ]


def format_labels(hierarchical, depth, LoD, labels_dir):
    logger.info('Formatting labels')
    labels = get_labels(hierarchical, depth, LoD, labels_dir)
    if depth > 0:
        labels = [
            (building, label)
            for building, label in labels
            if label != 'Unqualifiable'
        ]
    return zip(*labels)


def visualize_features(features, labels, depth, hier, **visualization_args):
    logger.info('Visualizing features...')
    if depth > 2 or hier is False:
        logger.error('Not yet implemented')
        raise NotImplementedError
    features = build_maniflod(
        **visualization_args['manifold']
    ).fit_transform(features)
    logger.debug('New transformed features: %s', features)
    logger.info('Fitted and transformed features.')

    features = dimension_reduction(
        **visualization_args['dimension_reduction']
    ).fit_transform(features)
    logger.debug('New reduced features: %s', features)
    logger.info('Fitted and reduced features.')

    features_per_errors = {
        cat: np.array(
            [
                features[idx]
                for idx, _
                in [
                    (idx, label)
                    for idx, label in enumerate(labels)
                    if label == cat
                ]
            ]
        )
        for cat in set(labels)
    }
    logger.debug('separated features: %s', features_per_errors)
    logger.info('Safely separated features by categories.')

    if (
        visualization_args['dimension_reduction']['parameters']
        ['n_components'] == 2
    ):
        fig, ax = plt.subplots()
    else:
        figure = plt.figure(1)
        ax = Axes3D(figure)

    visualize_categories(
        ax,
        features_per_errors,
        **visualization_args['style']
    )
    logger.info('Succesfully visualized categories')


def visualize_categories(ax, features_per_errors, **style_args):
    logger.info('Visualizing categories...')
    number_of_categories = len(features_per_errors)
    for color, marker, (label, features) in zip(
        style_args['colors'][:number_of_categories],
        style_args['markers'][:number_of_categories],
        features_per_errors.items()
    ):
        visualize_category(
            ax,
            color,
            marker,
            label,
            features
        )
    ax.legend()


def visualize_category(ax, color, marker, label, features):
    logger.info(
        'Visualizing category: %s with color %s and %s marker',
        label,
        color,
        marker
    )
    logger.info('Unpacking coordinates...')
    coordinates = zip(
        *[
            list(couple)
            for couple
            in list(
                features
            )
        ]
    )
    logger.debug('Unpacked %s coordinates: %s', label, coordinates)
    ax.scatter(*coordinates, label=label, c=color, marker=marker)


def build_maniflod(**manifold_args):
    logger.info('Building a manifold transformer...')
    if manifold_args['algorithm'] == 'PCA':
        return sklearn.decomposition.PCA(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] == 'KernelPCA':
        logger.info('Manifold %s build', manifold_args['algorithm'])
        return sklearn.decomposition.KernelPCA(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] == 'DictionaryLearning':
        logger.info('Manifold %s build', manifold_args['algorithm'])
        return sklearn.decomposition.DictionaryLearning(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] == 'ICA':
        logger.info('Manifold %s build', manifold_args['algorithm'])
        return sklearn.decomposition.FastICA(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] == 'FactorAnalys':
        logger.info('Manifold %s build', manifold_args['algorithm'])
        return sklearn.decomposition.FactorAnalys(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] == 'SpectralEmbedding':
        logger.info('Manifold %s build', manifold_args['algorithm'])
        return sklearn.manifold.SpectralEmbedding(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] == 'LocallyLinearEmbedding':
        logger.info('Manifold %s build', manifold_args['algorithm'])
        return sklearn.manifold.LocallyLinearEmbedding(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] == 'Isomap':
        logger.info('Manifold %s build', manifold_args['algorithm'])
        return sklearn.manifold.Isomap(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] == 'MDS':
        logger.info('Manifold %s build', manifold_args['algorithm'])
        return sklearn.manifold.MDS(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] == 'TSNE':
        logger.info('Manifold %s build', manifold_args['algorithm'])
        return sklearn.manifold.TSNE(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] == 'RBM':
        logger.info('Manifold %s build', manifold_args['algorithm'])
        return sklearn.neural_network.BernoulliRBM(
            **manifold_args['parameters']
        )
    else:
        logger.error(
            'Manifold %s is not supported.',
            manifold_args['algorithm']
        )
        raise LookupError


def dimension_reduction(**reductor_args):
    logger.info('Building a dimension reductor...')

    if reductor_args['parameters']['n_components'] > 3:
        logger.error('Cannot visualize more than three dimensions!')
        raise ValueError
    elif reductor_args['parameters']['n_components'] < 2:
        logger.error('Cannot visualize less than two dimensions!')
        raise ValueError

    if reductor_args['algorithm'] == 'PCA':
        logger.info('dimension reductor %s build', reductor_args['algorithm'])
        return sklearn.decomposition.PCA(**reductor_args['parameters'])
    else:
        logger.error(
            'Reductor %s is not supported.',
            reductor_args['algorithm']
        )
        raise LookupError


def classify(features, labels, **kwargs):
    logger.info('Classifying...')


def process(features, labels, depth, hierarchical, **kwargs):
    logger.info('Processing features...')
    if 'visualization' in kwargs.keys():
        logger.info('Feature space visualization.')
        visualize_features(
            features,
            labels,
            depth,
            hierarchical,
            **kwargs['visualization']
        )

    logger.info('Classification process')
    classify(features, labels, **kwargs['classification'])
    plt.show()


def load_pipeline_config(pip_conf):
    logger.info('Loading pipeline configuration file...')
    with open(pip_conf, mode='r') as conf:
        return yaml.load(conf)


def main():
    arguments = docopt.docopt(
        __doc__,
        help=True,
        version=None,
        options_first=False
    )

    config_logger(arguments)

    configuration = load_pipeline_config(arguments['<pipeline_config>'])
    logger.info('Pipeline loaded.')

    buildings, labels = format_labels(
        **configuration['labels']
    )
    logger.debug(
        'There are %s buildings. Buildings are: %s',
        len(buildings),
        buildings
    )
    logger.debug('Labels are: %s', labels)
    logger.info('Labels safely loaded.')

    features = format_features(
        configuration['labels']['depth'],
        buildings,
        configuration['features']
    )
    logger.debug(features)
    logger.info('Features safely loaded.')

    process(
        features,
        labels,
        configuration['labels']['depth'],
        configuration['labels']['hierarchical'],
        **configuration['processing']
    )


if __name__ == '__main__':
    main()

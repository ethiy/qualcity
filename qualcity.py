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

import yaml

import logging
import logging.config

import sklearn.decomposition
import sklearn.manifold

import sklearn.neural_network


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


def get_labels(hierarchical, depth, LoD, labels_dir):
    logger.info('Getting Labels ...')
    return labels_io.labels_map(
        labels_dir,
        hierarchical,
        depth,
        LoD,
        threshold=5
    )


def visualize_features(features, labels, **visualization_args):
    logger.info('Visualizing features...')
    features = build_maniflod(
        visualization_args['manifold']
    ).fit_transform(features)

    features = dimension_reduction(
        visualization_args['dimension_reduction']
    ).fit_transform(features)


def build_maniflod(**manifold_args):
    logger.info('Building a manifold transformer...')
    if manifold_args['algorithm'] is 'PCA':
        return sklearn.decomposition.PCA(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] is 'KernelPCA':
        return sklearn.decomposition.KernelPCA()
    elif manifold_args['algorithm'] is 'DictionaryLearning':
        return sklearn.decomposition.DictionaryLearning(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] is 'ICA':
        return sklearn.decomposition.FastICA(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] is 'FactorAnalysis':
        return sklearn.decomposition.FactorAnalysis(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] is 'SpectralEmbedding':
        return sklearn.manifold.SpectralEmbedding(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] is 'LocallyLinearEmbedding':
        return sklearn.manifold.LocallyLinearEmbedding(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] is 'Isomap':
        return sklearn.manifold.Isomap(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] is 'MDS':
        return sklearn.manifold.MDS(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] is 'TSNE':
        return sklearn.manifold.TSNE(
            **manifold_args['parameters']
        )
    elif manifold_args['algorithm'] is 'RBM':
        return sklearn.neural_network.BernoulliRBM(
            **manifold_args['parameters']
        )
    else:
        logger.error('Manifold %s is not supported.', manifold)
        LookupError
    logger.info('Manifold %s build', manifold)


def dimension_reduction(reductor, **reductor_args):
    logger.info('Building a dimension reductor...')
    if reductor is 'PCA':
        return sklearn.decomposition.PCA(**reductor_args)
    else:
        logger.error('Reductor %s is not supported.', manifold)
        LookupError


def classify(features, labels, **kwargs):
    logger.info('Classifying...')


def process(features, labels, **kwargs):
    logger.info('Processing features...')
    if 'visualization' in kwargs.keys():
        logger.info('Feature space visualization.')
        visualize_features(features, labels, **kwargs['visualization'])

    logger.info('Classification process')
    classify(features, labels, **kwargs['classification'])


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

    process(features, labels, **configuration['processing'])


if __name__ == '__main__':
    main()

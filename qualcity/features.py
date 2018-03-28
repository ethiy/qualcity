# -*- coding: <utf-8> -*-

import logging

import ast

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import altimetric_features
from . import radiometric_features
from . import geometric_features

feature_logger = logging.getLogger(__name__)


def feature(feat_type, **kwargs):
    if feat_type == 'geometric':
        if 'paramaters' in kwargs.keys():
            return geometric_features.geometric_features(
                kwargs['graph_dir'],
                kwargs['attributes'],
                kwargs['statistics'],
                **kwargs['paramaters']
            )
        else:
            return geometric_features.geometric_features(
                kwargs['graph_dir'],
                kwargs['attributes'],
                kwargs['statistics']
            )
    elif feat_type == 'altimetric':
        if 'margins' in kwargs.keys():
            kwargs['margins'] = ast.literal_eval(kwargs['margins'])
        return altimetric_features.histogram_features(**kwargs)
    elif feat_type == 'radiometric':
        return radiometric_features.radiometric_features(**kwargs)
    else:
        raise NotImplementedError(
            'Attribute type {} not implemented'.format(feat_type)
        )


def get_features(buildings, **config):
    feature_logger.info('Getting features ...')
    feature_dicts = [
        feature(feat_type, **config[feat_type])
        for feat_type in config.keys()
    ]
    return [
        np.concatenate(
            [
                feature_dict[building]
                for feature_dict in feature_dicts
            ]
        )
        for building in buildings
    ]


def visualize_features(
    features,
    labels,
    label_names,
    embedding,
    reductor,
    visualization_dimension,
    **style_args
):
    feature_logger.info('Visualizing features...')
    try:
        if isinstance(label_names, list):
            for label_name, _labels in zip(label_names, zip(*labels)):
                visualize_features(
                    features,
                    _labels,
                    (None, label_name),
                    embedding,
                    reductor,
                    visualization_dimension,
                    **style_args
                )
        elif isinstance(label_names, tuple):
            features = embedding.fit_transform(features)
            feature_logger.debug('New transformed features: %s', features)
            feature_logger.info('Fitted and transformed features.')

            features = reductor.fit_transform(features)
            feature_logger.debug('New reduced features: %s', features)
            feature_logger.info('Fitted and reduced features.')

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
                for cat in ([0, 1] if None in label_names else label_names)
            }
            feature_logger.debug('separated features: %s', features_per_errors)
            feature_logger.info('Safely separated features by categories.')

            if visualization_dimension == 2:
                fig, ax = plt.subplots()
            else:
                figure = plt.figure()
                ax = Axes3D(figure)

            ax.set_title(
                ' '.join(
                    ['Feature visualization for']
                    + (
                        [label_names[1]]
                        if None in label_names else list(label_names)
                    )
                )
            )
            visualize_categories(
                ax,
                features_per_errors,
                **style_args
            )
        else:
            feature_logger.error('Not yet implemented')
            raise NotImplementedError('Not yet implemented')
    except NotImplementedError:
        feature_logger.warn('Skipped visualization')
    feature_logger.info('Succesfully visualized categories')


def visualize_categories(ax, features_per_errors, **style_args):
    feature_logger.info('Visualizing categories...')
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
    feature_logger.info(
        'Visualizing category: %s with color %s and %s marker',
        label,
        color,
        marker
    )
    feature_logger.info('Unpacking coordinates...')
    coordinates = zip(
        *[
            list(couple)
            for couple
            in list(
                features
            )
        ]
    )
    feature_logger.debug('Unpacked %s coordinates: %s', label, coordinates)
    ax.scatter(*coordinates, label=label, c=color, marker=marker)

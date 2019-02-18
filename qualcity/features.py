# -*- coding: <utf-8> -*-

import logging

import os
import uuid
import ast

import functools
import operator

import tqdm

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import altimetric_features
from . import radiometric_features
from . import geometric_features
from . import utils

feature_logger = logging.getLogger(__name__)


def get_modality_features(buildings, feat_type, cache_dir, **kwargs):
    if feat_type == 'geometric':
        features = geometric_features.geometric_features(buildings, **kwargs)
    elif feat_type == 'altimetric':
        features = altimetric_features.altimetric_features(buildings, cache_dir, **kwargs)        
    elif feat_type == 'radiometric':
        features = radiometric_features.radiometric_features(buildings, **kwargs)
    else:
        raise NotImplementedError(
            'Attribute type {} not implemented'.format(feat_type)
        )
    return features


def compute_features(buildings, cache_dir, **feature_types):
    feature_logger.info(
        'Fetching features of modalities %s in dictionnary...',
        feature_types.keys()
    )
    return {
        feat_type: get_modality_features(
            buildings,
            feat_type,
            cache_dir,
            **feature_types[feat_type]
        )
        for feat_type in feature_types.keys()
    }


def compute_kernel(features, **kernel_args):
    if 'algorithm' in kernel_args.keys():
        if 'classe' not in kernel_args['algorithm'].keys():
            return utils.resolve(kernel_args['algorithm'])(features,**kernel_args['parameters'])
        else:
            return utils.resolve(kernel_args['algorithm']['classe'])(**kernel_args['algorithm']['parameters']).fit_transform(features)
    else:
        return sum(
            [
                compute_kernel(
                    [
                        feature[kernel_format]
                        for feature in features
                    ],
                    **parameters
                )
                for (kernel_format, parameters) in kernel_args.items()
            ]
        )


def get_features(buildings, cache_dir, **feature_configs):
    feature_logger.info('Getting features...')
    modalities_features = compute_features(buildings, cache_dir, **feature_configs['types'])
    if list(feature_configs['format'].keys()) == ['vector']:
        return (
            'vector',
            [
                np.concatenate(
                    [
                        np.concatenate(
                            [
                                modality_feature[building]
                                for modality_feature in modality_features.values()
                            ]
                        )
                        for modality_features in modalities_features.values()
                    ]
                )
                for building in buildings
            ]
        )
    elif list(feature_configs['format'].keys()) == ['kernel']:
        return (
            'kernel',
            sum(
                [
                    compute_kernel(
                        [
                            modalities_features[feat_type][building]
                            for building in buildings
                        ],
                        **parameters
                    )
                    for (feat_type, parameters) in feature_configs['format']['kernel'].items()
                ]
            )
        )
    else:
        raise NotImplementedError('Unknown feature format')


def visualize_features(
    features,
    labels,
    label_names,
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
                    reductor,
                    visualization_dimension,
                    **style_args
                )
        elif isinstance(label_names, tuple):
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
    xs, ys, zs = zip(
        *[
            list(couple)
            for couple
            in list(
                features
            )
        ]
    )
    feature_logger.debug('Unpacked %s coordinates: %s', label, (xs, ys, zs))
    ax.scatter(xs, ys, zs, label=label, c=color, marker=marker)

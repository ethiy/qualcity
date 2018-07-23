# -*- coding: <utf-8> -*-

import logging

import os
import uuid
import ast

import tqdm

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import altimetric_features
from . import radiometric_features
from . import geometric_features
from . import utils

feature_logger = logging.getLogger(__name__)


def atributes(buildings, feat_type, cache_dir, **kwargs):
    ledger = utils.cache_ledger(cache_dir, 'features')
    cached_features = {
        building: utils.read_cached_feature(
            cache_dir,
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
                list(kwargs.items())
            ),
            ledger
        )
        for building in tqdm.tqdm(
            buildings,
            desc='Cached ' + feat_type + ' features'
        )
    }
    cached_features.update(
        compute_attributes(
            [
                building
                for building in buildings
                if cached_features[building] is None
            ],
            feat_type,
            cache_dir,
            **kwargs
        )
    )
    return cached_features


def compute_attributes(buildings, feat_type, cache_dir, **kwargs):
    if feat_type == 'geometric':
        if 'paramaters' in kwargs.keys():
            features = geometric_features.geometric_features(
                buildings,
                kwargs['graph_dir'],
                kwargs['attributes'],
                kwargs['statistics'],
                **kwargs['paramaters']
            )
        else:
            features = geometric_features.geometric_features(
                buildings,
                kwargs['graph_dir'],
                kwargs['attributes'],
                kwargs['statistics']
            )
    elif feat_type == 'altimetric':
        if 'margins' in kwargs.keys():
            kwargs['margins'] = ast.literal_eval(kwargs['margins'])
        features = altimetric_features.histogram_features(buildings, **kwargs)        
    elif feat_type == 'radiometric':
        features = radiometric_features.radiometric_features(buildings, **kwargs)
    else:
        raise NotImplementedError(
            'Attribute type {} not implemented'.format(feat_type)
        )
    utils.cache_features(cache_dir, feat_type, kwargs, features)
    return features


def get_features(buildings, cache_dir, **feature_types):
    feature_logger.info('Getting features ...')
    feature_dicts = [
        atributes(buildings, feat_type, cache_dir, **feature_types[feat_type])
        for feat_type in feature_types.keys()
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


def build_maniflod(algorithm, **parameters):
    feature_logger.info('Building a manifold transformer...')
    return utils.resolve(algorithm)(
        **parameters
    )


def transform(features, **manifold_args):
    feature_logger.info(
        'Transforming features using %s...',
        manifold_args['algorithm']
    )
    return build_maniflod(
        manifold_args['algorithm'],
        **manifold_args['parameters']
    ).fit_transform(features)


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
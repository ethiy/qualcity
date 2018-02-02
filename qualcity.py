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
import itertools
import operator

import yaml

import logging
import logging.config

import sklearn.decomposition
import sklearn.manifold
import sklearn.preprocessing

import sklearn.metrics

import sklearn.neural_network

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import altimetric_difference
import geometry_io
import labels_io
import utils

FIGURE_COUNTER = 1

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
        'geometric': lambda kwargs: geometry_io.geometric_features(
            kwargs['graph_dir'],
            kwargs['attributes'],
            kwargs['statistics'],
            **kwargs['paramaters']
        ) if 'paramaters' in kwargs.keys() else geometry_io.geometric_features(
            kwargs['graph_dir'],
            kwargs['attributes'],
            kwargs['statistics']
        ),
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
        raise NotImplementedError
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
    try:
        if depth > 2 or hier is False:
            logger.error('Not yet implemented')
            raise NotImplementedError
        features = build_maniflod(
            visualization_args['manifold']['algorithm'],
            **visualization_args['manifold']['parameters']
        ).fit_transform(features)
        logger.debug('New transformed features: %s', features)
        logger.info('Fitted and transformed features.')

        features = build_reductor(
            visualization_args['dimension_reduction']['algorithm'],
            **visualization_args['dimension_reduction']['parameters']
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
            for cat in sorted(list(set(labels)), reverse=True)
        }
        logger.debug('separated features: %s', features_per_errors)
        logger.info('Safely separated features by categories.')

        if (
            visualization_args['dimension_reduction']['parameters']
            ['n_components'] == 2
        ):
            fig, ax = plt.subplots()
        else:
            global FIGURE_COUNTER
            figure = plt.figure(FIGURE_COUNTER)
            FIGURE_COUNTER += 1
            ax = Axes3D(figure)

        visualize_categories(
            ax,
            features_per_errors,
            **visualization_args['style']
        )
    except NotImplementedError:
        logger.warn('Skipped visualization')
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


def build_maniflod(algorithm, **parameters):
    logger.info('Building a manifold transformer...')
    return utils.resolve(algorithm)(
        **parameters
    )


def build_reductor(algorithm, **parameters):
    logger.info('Building a dimension reductor...')

    if parameters['n_components'] > 3:
        logger.error('Cannot visualize more than three dimensions!')
        raise ValueError
    elif parameters['n_components'] < 2:
        logger.error('Cannot visualize less than two dimensions!')
        raise ValueError

    return utils.resolve(algorithm)(
        **parameters
    )


def build_classifier(**classifier_args):
    logger.info('Building a classifier...')

    model = utils.resolve(
        classifier_args['algorithm']
    )(**classifier_args['parameters'])
    logger.info(
        'Constructed classifier: %s, with parameters %s',
        classifier_args['algorithm'],
        classifier_args['parameters']
    )
    if 'strategy' in classifier_args.keys():
        logger.info('Adding strategy: %s', classifier_args['strategy'])
        model = utils.resolve(classifier_args['strategy'])(model)
    return model


def classify(
    features,
    labels,
    buildings,
    depth,
    hierarchical,
    train_indices,
    test_indices,
    **class_args
):
    model = train(
        np.array(features)[train_indices],
        np.array(labels)[train_indices],
        depth,
        (
            None if hierarchical
            else (
                ['Building', 'Facet'] if depth < 3
                else labels_io.LABELS(2, ['Building', 'Facet'])
            )
        ),
        **class_args['training']
    )
    logger.info(
        'Succesfully trained %s on all the features.',
        class_args['training']['algorithm']
    )
    test(
        model,
        np.array(buildings)[test_indices],
        np.array(features)[test_indices],
        np.array(labels)[test_indices],
        (
            None if depth < 3
            else (
                None,
                labels_io.LABELS(2, ['Building']),
                labels_io.LABELS(2, ['Facet'])
            )
        ) if hierarchical
        else (
            ['Building', 'Facet'] if depth < 3
            else labels_io.LABELS(2, ['Building', 'Facet'])
        ),
        **class_args['testing']
    )
    logger.info(
        'Succesfully tested %s on all features.',
        class_args['training']['algorithm']
    )


def predict_proba(model, buildings, features, lnames=None, larray=True):
    logger.info('Predicting probabilities...')
    if lnames is None:
        try:
            logger.info('Multiclass predicition...')
            return {
                building: proba
                for building, proba
                in zip(
                    buildings,
                    np.amax(
                        model.predict_proba(features),
                        axis=1
                    )
                )
            }
        except AttributeError:
            logger.info('Multiclass, Multilabel stage predicition...')
            return {
                building: (
                        probability,
                        predict_proba(
                            model[family],
                            [building],
                            features[buildings.index(building)].reshape(1, -1),
                            labels_io.LABELS(2, family)
                        )[building] if family != 'Valid' else []
                    )
                for building, family, probability
                in zip(
                    buildings,
                    model[None].predict(features),
                    np.amax(
                        model[None].predict_proba(features),
                        axis=1
                    )
                )
            }
    else:
        logger.info('Multilabel stage predicition...')
        return {
            building: probabilties
            for building, probabilties
            in zip(
                buildings,
                model.predict_proba(features)
            )
        }


def predict(model, buildings, features, lnames=None, larray=True):
    logger.info('Predicting...')
    if lnames is None:
        logger.info('Multiclass predicition...')
        return {
            building: cls
            for building, cls
            in zip(
                buildings,
                model.predict(features)
            )
        }
    elif isinstance(lnames, tuple):
        logger.info('Multiclass, Multilabel stage predicition...')
        return {
            building:
            (
                family,
                (
                    predict(
                        model[family],
                        [building],
                        features[buildings.index(building)].reshape(1, -1),
                        labels_io.LABELS(2, family),
                        larray=True
                    )[building] if family != 'Valid' else None
                )
            )
            for building, family
            in zip(
                buildings,
                model[None].predict(features)
            )
        }
    else:
        logger.info('Multilabel stage predicition...')
        return {
            building: list(labels)
            for building, labels
            in zip(
                buildings,
                model.predict(features)
            )
        }


def test(model, buildings, features, ground_truth, label_names, **test_args):
    predictions = predict(
        model,
        buildings,
        features,
        label_names,
        test_args['probabilties']
    )

    cm = report(
        [
            predictions[building]
            for building in buildings
        ],
        ground_truth,
        label_names,
        *test_args['score']
    )

    print(cm)

    save_prediction(
        predictions,
        test_args['filename']
    )


def train(features, true, depth, multilabels=None, **train_args):
    logger.info('Training...')
    model = build_classifier(**train_args)

    if multilabels is None:
        if depth < 3:
            logger.info('Fitting and predicting classes...')
            predicted = model.fit(features, np.array(true)).predict(
                features
            )
            if train_args['reporting']:
                logger.info(
                    'Reporting classes %s ...',
                    set(labels)
                )
                report(predicted, true)
            return model
        else:
            logger.info('Separate families from classes...')
            families, errors = zip(*true)
            logger.info('Fitting and predicting families...')
            predicted_families = model.fit(
                features, np.array(families)
            ).predict(features)
            if train_args['reporting']:
                logger.info(
                    'Reporting classes %s ...',
                    set(families)
                )
                report(predicted_families, families)
            logger.info('Separating errors per family...')
            idx_per_fam = {
                fam: np.array(
                    [
                        idx
                        for idx, _
                        in [
                            (idx, label)
                            for idx, label in enumerate(families)
                            if label == fam
                        ]
                    ]
                )
                for fam in set(families)
                if fam != 'Valid'
            }

            return dict(
                [(None,  model)]
                +
                [
                    (
                        fam,
                        train(
                            [features[idx] for idx in fam_indexes],
                            [errors[idx] for idx in fam_indexes],
                            0,
                            labels_io.LABELS(2, fam),
                            **train_args
                        )
                    )
                    for fam, fam_indexes in idx_per_fam.items()
                ]
            )
    else:
        logger.info('Fitting and predicting multilabels...')
        predicted = model.fit(features, np.array(true)).predict(
            features
        )
        if train_args['reporting']:
            logger.info(
                'Reporting multilabels %s ...',
                multilabels
            )
            report(predicted, true, multilabels)
        return model


def save_prediction(predictions, filename):
    with open(filename, 'w') as prediction_file:
        for building, labels in predictions.items():
            prediction_file.write(
                ', '.join(
                    [building]
                    +
                    [
                        (
                            ', '.join(label)
                            if isinstance(label, list)
                            else str(label)
                        )
                        for label in labels
                    ]
                )
                +
                '\n'
            )


def report(predicted, true, labels=None, *score_args):
    if isinstance(labels, tuple):
        (
            (true_families, _),
            (predicted_families, _)
        ) = (
            zip(*true),
            zip(*predicted)
        )
        return (
            [
                (
                    None,
                    report(
                        predicted_families,
                        true_families,
                        None,
                        *score_args
                    )
                )
            ]
            +
            [
                (
                    family,
                    report(
                        z_predicted_errors,
                        z_true_errors,
                        labels_io.LABELS(2, [family]),
                        *score_args
                    )
                )
                for family, (z_predicted_errors, z_true_errors)
                in [
                    (
                        family,
                        zip(
                            *[
                                (predicted_errors, true_errors)
                                for (true_family, true_errors),
                                    (predicted_family, predicted_errors)
                                in zip(
                                    true,
                                    predicted
                                )
                                if family == predicted_family
                                if true_family == predicted_family
                            ]
                        )
                    )
                    for family in set(predicted_families)
                    if family != 'Valid'
                ]
            ]
        )
    elif isinstance(labels, list):
        return [
            report(z_predicted, z_true, labels[number], *score_args)
            for number, (z_predicted, z_true) in enumerate(
                zip(zip(*predicted), zip(*true))
            )
        ]
    else:
        return sklearn.metrics.confusion_matrix(
            true,
            predicted,
            labels=['None', labels] if labels is not None else labels
        )


def plot_confusion_matrix(
    confusion_matrix,
    classes,
    figure,
    ax,
    normalize=False
):
    logger.info('Plotting confusion matrix...')
    logger.debug('Confusiong matrix is: %s', confusion_matrix)
    number_of_elements = np.sum(confusion_matrix)
    if normalize:
        logger.info('Normalizing confusion matrix')
        confusion_matrix = (
            confusion_matrix.astype('float')
            /
            confusion_matrix.sum(axis=1)[:, np.newaxis]
        )

    logger.debug('Confusion matrix to be plotted: %s', confusion_matrix)

    logger.info('Plotting now...')
    cm = ax.imshow(
        confusion_matrix,
        interpolation='nearest',
        cmap=plt.cm.Blues
    )
    ax.set_title(
        'Confusion matrix for '
        +
        str(number_of_elements)
        +
        ' elements'
    )
    figure.colorbar(cm, ax=ax)

    middle = confusion_matrix.max() / 2.
    for i, j in itertools.product(
        range(confusion_matrix.shape[0]),
        range(confusion_matrix.shape[1])
    ):
        plt.text(
            j,
            i,
            format(confusion_matrix[i, j], '.2f' if normalize else 'd'),
            horizontalalignment="center",
            color="white" if confusion_matrix[i, j] > middle else "black"
        )

    figure.tight_layout()
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xticklabels(classes)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')


def data_split(features, labels, **separation_args):
    if separation_args == 'None':
        return (
            np.arange(features.shape[0]),
            np.arange(features.shape[0])
        )
    elif 'train_test_split' in separation_args.keys():
        return sklearn.model_selection.train_test_split(
            np.arange(features.shape[0]),
            **separation_args['train_test_split']
        )
    elif 'cross_validation' in separation_args.keys():
        return list(
            sklearn.model_selection.StratifiedKFold(
                **separation_args['cross_validation']['parameters']
            ).split(
                features,
                list(zip(*labels))[0] if isinstance(labels, tuple) else labels
            )
        )
    else:
        logger.error(
            'Separation %s not implemented.',
            kwargs['classification']['data_separation']
        )
        raise NotImplementedError


def process(features, labels, buildings, depth, hierarchical, **kwargs):
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

    logger.info('Classification process starting...')
    indices = data_split(
        features,
        labels,
        **kwargs['classification']['data_separation']
    )
    if isinstance(indices, tuple):
        classify(
            features,
            labels,
            buildings,
            depth,
            hierarchical,
            indices[0],
            indices[1],
            **kwargs['classification']
        )
    else:
        for train_indices, test_indices in indices:
            classify(
                features,
                labels,
                buildings,
                depth,
                hierarchical,
                train_indices,
                test_indices,
                **kwargs['classification']
            )
    logger.info('Succesfully classified features.')

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
        buildings,
        configuration['labels']['depth'],
        configuration['labels']['hierarchical'],
        **configuration['processing']
    )


if __name__ == '__main__':
    main()

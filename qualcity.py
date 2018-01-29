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
            **visualization_args['manifold']
        ).fit_transform(features)
        logger.debug('New transformed features: %s', features)
        logger.info('Fitted and transformed features.')

        features = build_reductor(
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


def build_maniflod(**manifold_args):
    logger.info('Building a manifold transformer...')
    return utils.resolve(manifold_args['algorithm'])(
        **manifold_args['parameters']
    )


def build_reductor(**reductor_args):
    logger.info('Building a dimension reductor...')

    if reductor_args['parameters']['n_components'] > 3:
        logger.error('Cannot visualize more than three dimensions!')
        raise ValueError
    elif reductor_args['parameters']['n_components'] < 2:
        logger.error('Cannot visualize less than two dimensions!')
        raise ValueError

    return utils.resolve(reductor_args['algorithm'])(
        **reductor_args['parameters']
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


def classify(features, labels, buildings, depth, hierarchical, **class_args):
    if len(class_args.keys()) > 1:
        logger.warn('There more than one classification task!')
    elif len(class_args.keys()) == 0:
        logger.warn('There is no classification task!')
    else:
        if 'training' in class_args.keys():
            model = train(
                features,
                labels,
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
                'Succesfully trained %s on features.',
                class_args['training']['algorithm']
            )
            predictions = predict_all(
                model,
                buildings,
                features,
                (
                    None if hierarchical
                    else (
                        ['Building', 'Facet'] if depth < 3
                        else labels_io.LABELS(2, ['Building', 'Facet'])
                    )
                )
            )

            report_prediction(
                predictions,
                str('multilabel_'if not hierarchical else '')
                +
                'results'
                +
                '.csv'
            )
        else:
            logger.error(
                '%s Not yet implemented!',
                class_args.keys()
            )
            raise NotImplementedError


def predict_all(model, buildings, features, label_names=None):
    logger.info('Testing...')
    if label_names is None:
        try:
            return {
                building: (cls, proba)
                for building, cls, proba
                in zip(
                    buildings,
                    model.predict(features),
                    np.amax(
                        model.predict_proba(features),
                        axis=1
                    )
                )
            }
        except AttributeError:
            return {
                building: [
                    family,
                    probability,
                ]
                +
                (
                    predict_all(
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
        return {
            building: [
                el
                for tup in [
                    (label_name, bool(label), probability)
                    for label_name, label, probability
                    in zip(
                        label_names,
                        labels,
                        probabilties
                    )
                ]
                for el in tup
            ]
            for building, labels, probabilties
            in zip(
                buildings,
                model.predict(features),
                model.predict_proba(features)
            )
        }


def train(features, true, depth, multilabels=None, **kwargs):
    logger.info('Training...')
    model = build_classifier(**kwargs)

    if multilabels is None:
        if depth < 3:
            logger.info('Fitting and predicting classes...')
            predicted = model.fit(features, np.array(true)).predict(
                features
            )
            report_training(predicted, true)
            return model
        else:
            logger.info('Separate families from classes...')
            families, errors = zip(*true)
            logger.info('Fitting and predicting families...')
            predicted_families = model.fit(
                features, np.array(families)
            ).predict(features)
            report_training(predicted_families, families)
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
                            **kwargs
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
        report_multilabel_training(predicted, true, multilabels)
        return model


def report_multilabel_training(z_predicted, z_true, labels):
    logger.info('Reporting a multilabel training...')
    for number, (predicted, true) in enumerate(
        zip(zip(*z_predicted), zip(*z_true))
    ):
        logger.info(
            'Reporting label: %s', labels[number]
        )
        report_training(predicted, true, labels[number])


def report_prediction(predictions, filename):
    with open(filename, 'w') as prediction_file:
        for building, labels in predictions.items():
            prediction_file.write(
                building
                +
                ', '
                +
                functools.reduce(
                    lambda x, y: str(x) + ', ' + str(y),
                    labels
                )
                +
                '\n'
            )


def report_training(predicted, true, label=None):
    logger.info('Reporting training...')
    print(sklearn.metrics.classification_report(true, predicted))
    logger.debug('%s', sklearn.metrics.classification_report(true, predicted))
    global FIGURE_COUNTER
    f, ax = plt.subplots()
    FIGURE_COUNTER += 1
    plot_confusion_matrix(
        sklearn.metrics.confusion_matrix(true, predicted),
        (
            sorted(list(set(true)))
            if label is None
            else ['None', label]
        ),
        f,
        ax,
        normalize=False
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
    classify(
        features,
        labels,
        buildings,
        depth,
        hierarchical,
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

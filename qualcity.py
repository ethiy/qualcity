#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

"""qualcity.

Usage:
    qualcity.py (-h | --help)
    qualcity.py pipeline <pipline_conf> [logger <log_conf>] [(-v | --verbose)]

Options:
    -h --help           Show this screen.
    -v --verbose        Verbose mode

"""

import docopt

import time

import functools
import itertools
import operator

import pathos.multiprocessing as mp

import yaml

import logging
import logging.config

import sklearn.decomposition
import sklearn.manifold
import sklearn.preprocessing

import sklearn.externals.joblib as skPickler

import sklearn.metrics

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
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
        with open(arguments['<log_conf>'], mode='r') as conf:
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
        logger.info('Loaded logger from: ' + arguments['<log_conf>'])
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


def get_labels(hierarchical, depth, LoD, threshold, labels_path, filetype):
    logger.info('Getting Labels ...')
    return [
        (building, label)
        for building, label in sorted(
            labels_io.labels_map(
                labels_path,
                hierarchical,
                depth,
                LoD,
                threshold,
                filetype
            ).items(),
            key=operator.itemgetter(0)
        )
    ]


def format_labels(hierarchical, depth, LoD, threshold, labels_path, filetype):
    logger.info('Formatting labels')
    labels = get_labels(
        hierarchical,
        depth,
        LoD,
        threshold,
        labels_path,
        filetype
    )
    if depth > 0:
        labels = [
            (building, label)
            for building, label in labels
            if label != 'Unqualifiable'
        ]
    return zip(*labels)


def visualize_features(features, labels, label_names, **visualization_args):
    logger.info('Visualizing features...')
    try:
        if not isinstance(label_names, tuple):
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

    if 'filename' in classifier_args:
        return skPickler.load(classifier_args['filename'])
    else:
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


def train_test(
    features,
    labels,
    buildings,
    label_names,
    train_indices,
    test_indices,
    **class_args
):
    model, train_cm = train(
        np.array(features)[train_indices],
        None if labels is None else [labels[idx] for idx in train_indices],
        label_names,
        **class_args['training']
    )
    if 'save' in class_args['training']['model']:
        skPickler.dump(model, class_args['training']['model']['save'] + '.pkl')
    logger.info(
        'Succesfully trained on all the features.'
    )
    predictions, test_cm = test(
        model,
        [buildings[idx] for idx in test_indices],
        np.array(features)[test_indices],
        None if labels is None else [labels[idx] for idx in test_indices],
        label_names,
        **class_args['testing']
    )
    logger.info(
        'Succesfully tested on all features.'
    )

    return (
        predictions,
        predict_proba(
            model,
            [buildings[index] for index in test_indices],
            np.array(features)[test_indices],
            label_names
        ),
        train_cm,
        test_cm
    )


def summarize_cv(cms, label_names, train=None, cv=1):
    if isinstance(label_names, dict):
        summed_dict = functools.reduce(
            lambda ldict, rdict: {
                fam: (
                    (
                        [sum(lists) for lists in zip(llist, rdict[fam])]
                        if fam is not None
                        else llist + rdict[fam]
                    )
                    if fam in rdict else llist
                )
                for fam, llist in ldict.items()
            },
            cms,
            dict(
                [
                    (
                        None,
                        np.zeros(
                            (len(label_names.keys()), len(label_names.keys())),
                            dtype=int
                        )
                    )
                ] + [
                    (
                        fam,
                        [np.zeros((2, 2), dtype=int)] * len(label_names[fam])
                    )
                    for fam in label_names.keys()
                    if fam != 'Valid'
                ]
            )
        )
        return summed_dict if not train else {
            fam: cm // cv if fam is None else [
                _cm // cv for _cm in cm
            ]
            for fam, cm in summed_dict.items()
        }
    elif isinstance(label_names, list):
        return [
            cm // (cv if train else 1)
            for cm in functools.reduce(
                lambda llist, rlist: [
                    sum(lsts) for lsts in zip(llist, rlist)
                ],
                cms,
                [np.zeros((2, 2), dtype=int)] * len(label_names)
            )
        ]
    elif isinstance(label_names, tuple):
        return functools.reduce(
            operator.add,
            cms,
            np.zeros((len(label_names), len(label_names)), dtype=int)
        ) // (cv if train else 1)
    else:
        raise LookupError('Labels %s not supported', label_names)


def classify(features, labels, buildings, label_names, **class_args):
    logger.info('Classification process starting...')

    indices = data_split(
        features,
        labels,
        **(
            class_args['data_separation']
            if 'data_separation' in class_args else {}
        )
    )
    logger.info('Data splited.')

    predictions, proba_predictions, train_cm, test_cm = (
        None,
        None,
        None,
        None
    )
    if isinstance(indices, tuple):
        predictions, proba_predictions, train_cm, test_cm = train_test(
            features,
            labels,
            buildings,
            label_names,
            indices[0],
            indices[1],
            **class_args
        )
    elif isinstance(indices, list):
        pool = mp.Pool(processes=len(indices))
        z_predictions, z_proba_predictions, train_cms, test_cms = zip(
            *pool.map(
                lambda p: (
                    lambda train_idx, test_idx: train_test(
                        features,
                        labels,
                        buildings,
                        label_names,
                        train_idx,
                        test_idx,
                        **class_args
                    )
                )(*p),
                indices
            )
        )
        predictions = dict(
            functools.reduce(
                lambda llist, rdict: llist + list(rdict.items()),
                z_predictions,
                []
            )
        )
        proba_predictions = dict(
            functools.reduce(
                lambda llist, rdict: llist + list(rdict.items()),
                z_proba_predictions,
                []
            )
        )

        train_cm, test_cm = [
            summarize_cv(cms, label_names, train, cv)
            for train, cms, cv in zip(
                [True, False],
                [train_cms, test_cms],
                [len(indices) - 1, 1]
            )
        ]
    else:
        predictions, proba_predictions, train_cm, test_cm = train_test(
            features,
            None,
            buildings,
            label_names,
            [],
            indices,
            **class_args
        )
    save_prediction(
        predictions,
        proba_predictions,
        label_names,
        **class_args['testing']['predictions']
    )
    if train_cm is not None and test_cm is not None:
        for cm in [train_cm, test_cm]:
            plot_confusion_matrix(cm, label_names)


def predict_proba(model, buildings, features, label_names):
    logger.info('Predicting probabilities...')
    if isinstance(label_names, tuple):
        return {
            building: probability
            for building, probability
            in zip(
                buildings,
                np.amax(
                    model.predict_proba(features),
                    axis=1
                )
            )
        }
    elif isinstance(label_names, list):
        return {
            building: probabilities
            for building, probabilities
            in zip(
                buildings,
                model.predict_proba(features)
            )
        }
    elif isinstance(label_names, dict):
        return {
            building: [
                error_probablities * probability
                for error_probablities in predict_proba(
                    model[family],
                    [building],
                    features[buildings.index(building)].reshape(1, -1),
                    label_names[family]
                )[building]
            ] if family != 'Valid' else probability
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
        raise LookupError('Labels %s not supported', label_names)


def predict(model, buildings, features, label_names):
    logger.info('Predicting...')
    if isinstance(label_names, tuple):
        logger.info('Multiclass predicition...')
        return {
            building: cls
            for building, cls
            in zip(
                buildings,
                model.predict(features)
            )
        }
    elif isinstance(label_names, dict):
        logger.info('Multiclass, Multilabel stage predicition...')
        return {
            building:
            (
                family,
                (
                    predict(
                        model[family],
                        [building],
                        features[
                            buildings.index(building)
                        ].reshape(1, -1),
                        label_names[family]
                    )[building] if family != 'Valid' else None
                )
            )
            for building, family
            in zip(
                buildings,
                model[None].predict(features)
            )
        }
    elif isinstance(label_names, list):
        logger.info('Multilabel predicition...')
        return {
            building: list(labels)
            for building, labels
            in zip(
                buildings,
                model.predict(features)
            )
        }
    else:
        raise LookupError('Labels %s not supported', label_names)


def test(model, buildings, features, ground_truth, label_names, **test_args):
    predictions = predict(
        model,
        buildings,
        features,
        label_names
    )

    return (
        predictions,
        report(
            [
                predictions[building]
                for building in buildings
            ],
            ground_truth,
            label_names
        ) if ground_truth is not None else None
    )


def train(features, true, label_names, **train_args):
    logger.info('Training...')
    model = build_classifier(**train_args['model'])

    if true is None:
        return (model, None)
    elif isinstance(label_names, tuple):
        logger.info('Fitting and predicting classes...')
        predicted = model.fit(features, np.array(true)).predict(
            features
        )
        return (model, report(predicted, true, label_names))
    elif isinstance(label_names, list):
        logger.info('Fitting and predicting multilabels...')
        predicted = model.fit(features, np.array(true)).predict(
            np.array(features)
        )
        return (model, report(predicted, true, label_names))
    elif isinstance(label_names, dict):
        logger.info('Separate families from classes...')
        families, errors = zip(*true)
        logger.info('Fitting and predicting families...')
        predicted_families = model.fit(
            features, np.array(families)
        ).predict(features)
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
            for fam in label_names.keys()
            if fam != 'Valid'
        }

        fams, models, reports = zip(
            *[
                (
                    fam,
                    *train(
                        [features[idx] for idx in fam_indexes],
                        [errors[idx] for idx in fam_indexes],
                        label_names[fam],
                        **train_args
                    )
                )
                for fam, fam_indexes in idx_per_fam.items()
            ]
        )

        return (
            dict(
                [(None, model)] + [
                    (fam, modl) for fam, modl in zip(fams, models)
                ]
            ),
            dict(
                [
                    (
                        None,
                        report(
                            predicted_families,
                            families,
                            tuple(label_names.keys())
                        )
                    )
                ]
                +
                [
                    (fam, report) for fam, report in zip(fams, reports)
                ]
            )
        )
    else:
        raise LookupError('Labels %s not supported', label_names)


def save_prediction(
    predictions,
    proba_predictions,
    label_names,
    filename='predictions'
):
    with open(filename + '.csv', 'w') as prediction_file:
        if isinstance(label_names, tuple):
            for building, labels in predictions.items():
                prediction_file.write(
                    ', '.join(
                        [
                            building,
                            labels,
                            '{:.3f}'.format(proba_predictions[building])
                        ]
                    )
                    +
                    '\n'
                )
        elif isinstance(label_names, list):
            for building, labels in predictions.items():
                prediction_file.write(
                    ', '.join(
                        [building] + list(
                            sum(
                                [
                                    (
                                        label_names[idx],
                                        '{:.3f}'.format(proba)
                                    )
                                    for idx, label, proba in zip(
                                        itertools.count(),
                                        labels,
                                        proba_predictions[building]
                                    )
                                    if label
                                ],
                                ()
                            )
                        ) if sum(labels) else [
                            building,
                            'Valid',
                            '{:.3f}'.format(
                                np.multiply.reduce(
                                    1 - proba_predictions[building]
                                )
                            )
                        ]
                    )
                    +
                    '\n'
                )
        elif isinstance(label_names, dict):
            for building, (family, errors) in predictions.items():
                prediction_file.write(
                    ', '.join(
                        [building] + list(
                            sum(
                                [
                                    (
                                        label_names[family][idx],
                                        '{:.3f}'.format(proba)
                                    )
                                    for idx, label, proba in zip(
                                        itertools.count(),
                                        errors,
                                        proba_predictions[building]
                                    )
                                    if label
                                ],
                                ()
                            )
                        )
                        if family != 'Valid' else [
                            building,
                            'Valid',
                            '{:.3f}'.format(proba_predictions[building])
                        ]
                    )
                    +
                    '\n'
                )
        else:
            raise LookupError('Labels %s not supported', label_names)


def report(predicted, true, label_names):
    if isinstance(label_names, dict):
        (
            (true_families, _),
            (predicted_families, _)
        ) = (
            zip(*true),
            zip(*predicted)
        )
        return dict(
            [
                (
                    None,
                    report(
                        predicted_families,
                        true_families,
                        tuple(label_names.keys())
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
                        label_names[family]
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
                    for family in predicted_families
                    if family != 'Valid'
                ]
            ]
        )
    elif isinstance(label_names, list):
        return [
            report(
                z_predicted,
                z_true,
                ('None', label_names[number])
            )
            for number, (z_predicted, z_true) in enumerate(
                zip(zip(*predicted), zip(*true))
            )
        ]
    elif isinstance(label_names, tuple):
        present = set(true)
        if len(present) != len(label_names):
            if len(present) == 1:
                cm = np.zeros((len(label_names), len(label_names)), dtype=int)
                if 1 in present:
                    cm[1, 1] = sklearn.metrics.confusion_matrix(
                        true,
                        predicted
                    )[0, 0]
                    return cm
                elif 0 in present:
                    cm[0, 0] = sklearn.metrics.confusion_matrix(
                        true,
                        predicted
                    )[0, 0]
                    return cm
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            return sklearn.metrics.confusion_matrix(
                true,
                predicted
            )
    else:
        raise LookupError('Labels %s not supported', label_names)


def plot_confusion_matrix(
    confusion_matrix,
    label_names,
    figure=None,
    axes=None,
    normalize=False,
    max_l=8
):
    if isinstance(label_names, dict):
        figure = plt.figure()
        specs = matplotlib.gridspec.GridSpec(
            len(confusion_matrix),
            max(
                [
                    len(matrices)
                    for matrices in confusion_matrix.values()
                ]
            )
        )
        plot_confusion_matrix(
            confusion_matrix[None],
            tuple(label_names.keys()),
            figure,
            figure.add_subplot(specs[0, 0]),
            normalize,
            max_l
        )
        confusion_matrix.pop(None)
        for row, (family, cms) in enumerate(confusion_matrix.items(), start=1):
            plot_confusion_matrix(
                cms,
                label_names[family],
                figure,
                [
                    figure.add_subplot(specs[row, column])
                    for column in range(len(cms))
                ],
                normalize,
                max_l
            )
    elif isinstance(confusion_matrix, list):
        figure, axes = (
            plt.subplots(1, len(confusion_matrix))
            if axes is None else (figure, axes)
        )
        plt.subplots_adjust(left=.05, right=.95)
        for column, cm in enumerate(confusion_matrix):
            plot_confusion_matrix(
                cm,
                ('Valid', label_names[column]),
                figure,
                axes[column],
                normalize,
                max_l
            )
    else:
        figure, ax = plt.subplots() if axes is None else (figure, axes)
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
            str(number_of_elements)
            +
            ' elements'
        )

        middle = confusion_matrix.max() / 2.
        for i, j in itertools.product(
            range(confusion_matrix.shape[0]),
            range(confusion_matrix.shape[1])
        ):
            ax.text(
                j,
                i,
                format(confusion_matrix[i, j], '.2f' if normalize else 'd'),
                horizontalalignment="center",
                color="white" if confusion_matrix[i, j] > middle else "black"
            )

        ticks = [
            (
                ' '.join(
                    _tick[:len(_tick)//2 + 1]
                    +
                    ['\n']
                    +
                    _tick[len(_tick)//2 + 1:]
                )
                if isinstance(_tick, list) else _tick
            )
            for _tick in [
                label.split() if len(label) > max_l else label
                for label in label_names
            ]
        ]
        ax.set_yticks(np.arange(len(label_names)))
        ax.set_xticks(np.arange(len(label_names)))
        ax.set_yticklabels(ticks)
        ax.set_xticklabels(ticks)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')


def data_split(features, labels, **separation_args):
    if not separation_args:
        return np.arange(len(features))
    elif 'train_test_split' in separation_args:
        return tuple(
            sklearn.model_selection.train_test_split(
                np.arange(len(features)),
                **separation_args['train_test_split']['parameters']
            )
        )
    elif 'cross_validation' in separation_args:
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


def process(features, labels, buildings, label_names, **kwargs):
    logger.info('Processing features...')
    if 'visualization' in kwargs.keys():
        logger.info('Feature space visualization.')
        visualize_features(
            features,
            labels,
            label_names,
            **kwargs['visualization']
        )
        logger.info('Visualization process ended.')

    classify(
        features,
        labels,
        buildings,
        label_names,
        **kwargs['classification']
    )
    logger.info('Succesfully classified features.')
    plt.show()


def load_pipeline_config(pip_conf):
    logger.info('Loading pipeline configuration file...')
    with open(pip_conf, mode='r') as conf:
        return yaml.load(conf)


def label_names(config, labels):
    if config['depth'] < 2:
        return tuple(set(labels))
    elif config['depth'] == 2:
        return (
            tuple(set(labels)) if config['hierarchical']
            else ['Building', 'Facet']
        )
    elif config['depth'] == 3:
        return (
            {
                'Valid': None,
                'Building': labels_io.LABELS(
                    config['LoD'],
                    ['Building']
                ),
                'Facet': labels_io.LABELS(
                    config['LoD'],
                    ['Facet']
                )
            } if config['hierarchical']
            else labels_io.LABELS(
                config['LoD'],
                ['Building', 'Facet']
            )
        )
    else:
        raise LookupError('depth cannot be > 3')


def main():
    arguments = docopt.docopt(
        __doc__,
        help=True,
        version=None,
        options_first=False
    )

    config_logger(arguments)

    configuration = load_pipeline_config(arguments['<pipline_conf>'])
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
        label_names(
            configuration['labels'],
            labels
        ),
        **configuration['processing']
    )


if __name__ == '__main__':
    main()

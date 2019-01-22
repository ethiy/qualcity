# -*- coding: <utf-8> -*-

import functools
import itertools
import operator

import logging

import yaml

import os
import uuid
import time

import pathos.multiprocessing as mp

import numpy as np

import sklearn.decomposition
import sklearn.manifold
import sklearn.preprocessing

import sklearn.externals.joblib as skPickler

import sklearn.metrics

import matplotlib
import matplotlib.pyplot as plt

from . import utils


learning_logger = logging.getLogger(__name__)


def build_classifier(**classifier_args):
    learning_logger.info('Building a classifier...')

    if 'filename' in classifier_args:
        return skPickler.load(classifier_args['filename'])
    else:
        model = utils.resolve(
            classifier_args['algorithm']
        )(**classifier_args['parameters'])
        learning_logger.info(
            'Constructed classifier: %s, with parameters %s',
            classifier_args['algorithm'],
            classifier_args['parameters']
        )
        if 'strategy' in classifier_args.keys():
            learning_logger.info('Adding strategy: %s', classifier_args['strategy'])
            model = utils.resolve(classifier_args['strategy'])(model)
        return model


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
    elif 'bipartite' in separation_args:
        return (np.arange(separation_args['bipartite']['threshold']), np.arange(separation_args['bipartite']['threshold'], len(features)))
    else:
        learning_logger.error(
            'Separation %s not implemented.',
            separation_args['classification']['data_separation']
        )
        raise NotImplementedError


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


def predict_proba(model, buildings, features, label_names):
    learning_logger.info('Predicting probabilities...')
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
    learning_logger.info('Predicting...')
    if isinstance(label_names, tuple):
        learning_logger.info('Multiclass predicition...')
        return {
            building: cls
            for building, cls
            in zip(
                buildings,
                model.predict(features)
            )
        }
    elif isinstance(label_names, dict):
        learning_logger.info('Multiclass, Multilabel stage predicition...')
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
        learning_logger.info('Multilabel predicition...')
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


def test(model, buildings, features, ground_truth, label_names, reportname, **test_args):
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
            label_names,
            reportname
        ) if ground_truth is not None else None
    )


def train(features, true, label_names, reportname, **train_args):
    learning_logger.info('Training...')
    model = build_classifier(**train_args['model'])

    if true is None:
        return (model, None)
    elif isinstance(label_names, tuple):
        learning_logger.info('Fitting and predicting classes...')
        predicted = model.fit(features, np.array(true)).predict(
            features
        )
        return (model, report(predicted, true, label_names, reportname))
    elif isinstance(label_names, list):
        learning_logger.info('Fitting and predicting multilabels...')
        predicted = model.fit(features, np.array(true)).predict(
            np.array(features)
        )
        return (model, report(predicted, true, label_names, reportname))
    elif isinstance(label_names, dict):
        learning_logger.info('Separate families from classes...')
        families, errors = zip(*true)
        learning_logger.info('Fitting and predicting families...')
        predicted_families = model.fit(
            features, np.array(families)
        ).predict(features)
        learning_logger.info('Separating errors per family...')
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
                            tuple(label_names.keys()),
                            reportname
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
    cache_dir,
    config
):
    cache_id = str(uuid.uuid4())
    utils.store_cache_ledger(cache_dir, 'predictions', cache_id, config)

    with open(os.path.join(cache_dir, 'predictions', cache_id + '.csv'), 'w') as prediction_file:
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


def report(predicted, true, label_names, cachename):
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
                        tuple(label_names.keys()),
                        cachename
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
                        label_names[family],
                        cachename
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
                ('None', label_names[number]),
                cachename
            )
            for number, (z_predicted, z_true) in enumerate(
                zip(zip(*predicted), zip(*true))
            )
        ]
    elif isinstance(label_names, tuple):
        current_labels = set(true)
        cm = np.zeros((len(label_names), len(label_names)), dtype=int)
        if len(current_labels) != len(label_names):
            if len(current_labels) == 1:
                if 1 in current_labels:
                    cm[1, 1] = sklearn.metrics.confusion_matrix(
                        true,
                        predicted
                    )[0, 0]
                elif 0 in current_labels:
                    cm[0, 0] = sklearn.metrics.confusion_matrix(
                        true,
                        predicted
                    )[0, 0]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            cm = sklearn.metrics.confusion_matrix(
                true,
                predicted
            )
        with open(cachename, 'a+') as cachefile:
            cachefile.write('Confusion matrix for: ' + ', '.join(label_names) + '\n')
            cachefile.write(np.array2string(cm) + '\n')
        return cm
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
        learning_logger.info('Plotting confusion matrix...')
        learning_logger.debug('Confusiong matrix is: %s', confusion_matrix)
        number_of_elements = np.sum(confusion_matrix)
        if normalize:
            learning_logger.info('Normalizing confusion matrix')
            confusion_matrix = (
                confusion_matrix.astype('float')
                /
                confusion_matrix.sum(axis=1)[:, np.newaxis]
            )

        learning_logger.debug(
            'Confusion matrix to be plotted: %s',
            confusion_matrix
        )

        learning_logger.info('Plotting now...')
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
                    _tick[:len(_tick) // 2 + 1]
                    +
                    ['\n']
                    +
                    _tick[len(_tick) // 2 + 1:]
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


def train_test(
    features,
    labels,
    buildings,
    label_names,
    train_indices,
    test_indices,
    cache_dir,
    cache_config,
    reportname,
    **class_args
):
    with open(
            os.path.join(
                cache_dir,
                'reports',
                reportname
            ),
            'a+'
        ) as reportfile:
        reportfile.write('Experiment parameters:\n')
        reportfile.write(
            yaml.dump(
                list(cache_config.items())
                +
                list(class_args.items())
            )
        )
        reportfile.write('\nModel trained over: ')
        reportfile.write(
            ', '.join(
                [buildings[idx] for idx in train_indices]
            )
        )
    cachedname_ = utils.cached(
        dict(
            [
                (
                    'training samples',
                    [buildings[idx] for idx in train_indices]
                )
            ]
            +
            list(cache_config.items())
            +
            list(class_args.items())
        ),
        utils.cache_ledger(
            cache_dir,
            'classifiers'
        )
    )
    if len(cachedname_):
        model = build_classifier(
            filename = os.path.join(
                cache_dir,
                'classifiers',
                cachedname_[0] + '.pkl'
            )
        )
        preds = predict(
            model,
            [buildings[idx] for idx in train_indices],
            np.array(features)[train_indices],
            label_names
        )
        with open(
                os.path.join(
                    cache_dir,
                    'reports',
                    reportname
                ),
                'a+'
            ) as reportfile:
                reportfile.write('Training report: \n')
        train_cm = report(
            [preds[buildings[idx]] for idx in train_indices],
            None if labels is None else [labels[idx] for idx in train_indices],
            label_names,
            os.path.join(
                cache_dir,
                'reports',
                reportname
            )
        )
        learning_logger.info(
            'Succesfully retreived from cache.'
        )
    else:
        model, train_cm = train(
            np.array(features)[train_indices],
            None if labels is None else [labels[idx] for idx in train_indices],
            label_names,
            os.path.join(
                cache_dir,
                'reports',
                reportname
            ),
            **class_args['training']
        )
        cache_id = str(uuid.uuid4())
        skPickler.dump(model, os.path.join(cache_dir, 'classifiers', cache_id + '.pkl'))
        utils.store_cache_ledger(
            cache_dir,
            'classifiers',
            cache_id,
            dict(
                [
                    (
                        'training samples',
                        [buildings[idx] for idx in train_indices]
                    )
                ]
                +
                list(cache_config.items())
                +
                list(class_args.items())
            )
        )
        learning_logger.info(
            'Succesfully trained on all the features.'
        )
    with open(
            os.path.join(
                cache_dir,
                'reports',
                reportname
            ),
            'a+'
        ) as reportfile:
        reportfile.write('\n\nTest report:\n')
    predictions, test_cm = test(
        model,
        [buildings[idx] for idx in test_indices],
        np.array(features)[test_indices],
        None if labels is None else [labels[idx] for idx in test_indices],
        label_names,
        os.path.join(
            cache_dir,
            'reports',
            reportname
        ),
        **class_args['testing']
    )
    learning_logger.info(
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


def classify(features, labels, buildings, label_names, cache_dir, cache_config, **class_args):
    learning_logger.info('Classification process starting...')
    reportname = 'report-' + time.ctime()
    indices = data_split(
        features,
        labels,
        **(
            class_args['data_separation']
            if 'data_separation' in class_args else {}
        )
    )
    learning_logger.info('Data splited.')

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
            cache_dir,
            cache_config,
            reportname,
            **class_args
        )
    elif isinstance(indices, list):
        pool = mp.Pool(processes=len(indices))
        z_predictions, z_proba_predictions, train_cms, test_cms = zip(
            *pool.map(
                lambda p: (
                    lambda cv_idx, idxes: train_test(
                        features,
                        labels,
                        buildings,
                        label_names,
                        idxes[0],
                        idxes[1],
                        cache_dir,
                        cache_config,
                        reportname + '-' + str(cv_idx) + '.txt',
                        **class_args
                    )
                )(*p),
                enumerate(indices)
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
        cache_dir,
        dict(
            list(cache_config.items())
            +
            list(class_args.items())
        )
    )
    if train_cm is not None and test_cm is not None:
        for cm in [train_cm, test_cm]:
            plot_confusion_matrix(cm, label_names)

#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import operator
import functools

from tqdm import tqdm

import shapefile

import numpy as np

from . import utils

label_logger = logging.getLogger(__name__)


SHAPEFILE = "ESRI Shapefile"

ERROR_DICTIONARY = {
    'Over Segmentation': [
        'Over Segmentation',
        'over_segmentation',
        'over',
        'over_segmantation',
        'over_segermentation',
        'oversegmentation',
        'over_segmentationover_segmentation'
    ],
    'Under Segmentation': ['Under Segmentation', 'under_segmentation', 'under', 'unde_segmentation'],
    'Imprecise Segmentation': ['Imprecise Segmentation', 'mis_segmentation', 'missegmentation', 'mis'],
    'Slope': ['Slope', 'slope'],
    'Footprint': ['Footprint', 'footprint', 'footprint_error'],
    'Height': ['too_low, Height'],
    'Valid': ['Valid', 'None', ''],
    'Unqualifiable': ['Unqualifiable', 'Unqualified']
}

UNQUALIFIABLE_ERROR_LIST = ['Unqualifiable']

BUILDING_ERROR_LIST = [
    'Building over segmentation',
    'Building under segmentation',
    'Building footprint imprecise borders',
    'Building footprint inaccurate topology',
    'Building imprecise geometry'
]

FACET_ERROR_LIST = [
    'Facet over segmentation',
    'Facet under segmentation',
    'Facet inaccurate topology',
    'Facet imprecise borders',
    'Facet imprecise geometry'
]

ERROR_CATEGORY_DICTIONARY = {
    'Unqualifiable': UNQUALIFIABLE_ERROR_LIST,
    'Building': BUILDING_ERROR_LIST,
    'Facet': FACET_ERROR_LIST
}

ERROR_CATEGORY_INDEX = {
    'Unqualifiable': 0,
    'Building': 1,
    'Facet': 2
}

CLASSES = {
    0: 'Valid',
    1: 'Unqualifiable',
    2: 'Building',
    3: 'Facet'
}

INV_CLASSES = {v: k for k, v in CLASSES.items()}


def LABELS(LoD, family):
    return (
        (LoD > 0) * ('Building' in family) * BUILDING_ERROR_LIST
        +
        (LoD > 1) * ('Facet' in family) * FACET_ERROR_LIST
    )


def labels_map(
    labels_path,
    hierarchical,
    depth,
    LoD,
    threshold,
    filetype='csv'
):
    if utils.caseless_equal(filetype, 'ESRI Shapefile'):
        return {
            os.path.splitext(shape)[0]: errors(
                read(os.path.join(labels_path, shape), filetype),
                hierarchical,
                depth,
                LoD,
                threshold
            )
            for shape in fnmatch.filter(
                os.listdir(labels_path),
                '*.shp'
            )
        }
    elif utils.caseless_equal(filetype, 'csv'):
        return {
            building: errors(
                error_dicts,
                hierarchical,
                depth,
                LoD,
                threshold
            )
            for building, error_dicts in tqdm(
                read(labels_path, filetype).items(),
                desc='Ground truth parsing'
            )
        }


def entry(error):
    label_logger.debug('Get error %s correct form', error)
    for _error, synonyms in ERROR_DICTIONARY.items():
        if error in synonyms:
            return _error
    raise LookupError('%s not found', error)


def set_score(error):
    label_logger.debug('Score of raw error %s', error)
    raw = error.split(': ')
    if len(raw) == 2:
        return (raw[0], int(raw[1]))
    elif len(raw) == 1:
        return (raw[0], 10)
    else:
        raise IOError('Cannot extract error score from %s', error)


def list_errors(errors):
    label_logger.debug('List errors in %s', errors)
    return [set_score(error) for error in errors.split(', ')]


def get_errors(feature):
    label_logger.debug('Raw errors from feature %s', feature)
    return (
        list_errors(feature[-3]),
        list_errors(feature[-2]),
        list_errors(feature[-1])
    )


def read_shp(filename):
    label_logger.info('Extracting labels from shapefile %s', filename)
    records = shapefile.Reader(filename).records()
    label_logger.debug('Records in %s are: %s', filename, records)
    unformatted_errors = unify_errors(
        functools.reduce(
            lambda lhs, rhs: [llhs + rrhs for llhs, rrhs in zip(lhs, rhs)],
            [get_errors(feature) for feature in records]
        )
    )

    return [
        {
            error:
            functools.reduce(
                max,
                [
                    _unified_errors[synonym]
                    for synonym in ERROR_DICTIONARY[error]
                    if synonym in _unified_errors.keys()
                ]
            )
            for error in set(
                [entry(key) for key in _unified_errors.keys()]
            )
        } if _unified_errors != 'Valid' else 'Valid'
        for _unified_errors in unformatted_errors
    ]


def read_csv(filename):
    possible_errors = UNQUALIFIABLE_ERROR_LIST + LABELS(2, ['Building', 'Facet'])
    milestones = np.cumsum(
        [0] + [len(l) for l in [UNQUALIFIABLE_ERROR_LIST, LABELS(1, ['Building'])]]
    )
    with open(filename, 'r') as label_file:
        raw_error_dict = {
            building: zip(
                [
                    possible_errors.index(label) - milestones
                    if label != 'Valid' else None
                    for label in labels[::3]
                ],
                [int(float(score)) for score in labels[2::3]]
            )
            for building, *labels in [
                line.split('\n')[0].split(',') for line in label_file.readlines()
                if len(line.split('\n')[0])
            ]
        }
    error_category_dict = {
        building:
        [
            (sum(indexes >= 0) - 1) * [{}]
            + [
                {
                    ERROR_CATEGORY_DICTIONARY[CLASSES[sum(indexes >= 0)]][
                        indexes[sum(indexes >= 0) - 1]
                    ]:
                    score
                }
            ]
            + (len(milestones) - sum(indexes >= 0)) * [{}]
            if indexes is not None else [{}] * len(milestones)
            for indexes, score in error_list
        ]
        for building, error_list in raw_error_dict.items()
    }
    return {
        building:
        [
            'Valid' if not len(level) else level
            for level in functools.reduce(
                lambda l_dicts, r_dicts:
                [
                    {**l_dict, **r_dicts[idx]}
                    for idx, l_dict in enumerate(l_dicts)
                ],
                error_dicts,
                [{}, {}, {}]
            )
        ]
        for building, error_dicts in error_category_dict.items()
    }


def read(filename, filetype='ESRI Shapefile'):
    if utils.caseless_equal(filetype, 'ESRI Shapefile'):
        return read_shp(filename)
    elif utils.caseless_equal(filetype, 'csv'):
        return read_csv(filename)
    else:
        raise LookupError('Unsupported filetype %s', filetype)


def write_csv(filename, errors_per_building):
    with open(filename, 'w') as label_file:
        for building, errors in errors_per_building.items():
            labels = list(
                sum(
                    [
                        error_score_pair
                        for level_errs in [
                            [
                                (
                                    (
                                        'Building ' + error
                                        if (
                                            level == 1
                                            and 'segmentation' in error
                                        )
                                        else (
                                            'Facet ' + error
                                            if (
                                                level == 2
                                                and 'r segmentation' in error
                                            )
                                            else error
                                        )
                                    ),
                                    '{:.3f}'.format(score / 10)
                                )
                                for error, score in level_errors.items()
                            ]
                            if isinstance(level_errors, dict)
                            else []
                            for level, level_errors in enumerate(errors)
                        ]
                        for error_score_pair in level_errs
                    ],
                    ()
                )
            )
            label_file.write(
                ', '.join(
                    [building]
                    +
                    (labels if labels else ['Valid', '1.000'])
                )
                + '\n'
            )


def write(filename, errors, filetype='csv'):
    if utils.caseless_equal(filetype, 'csv'):
        write_csv(filename, errors)
    else:
        raise LookupError('Unsupported filetype %s', filetype)


def lint(errors):
    label_logger.debug('Linting errors %s', errors)
    linted = {
        error: score
        for error, score in errors.items()
        if error not in ERROR_DICTIONARY['Valid']
    }
    return linted if linted != {} else 'Valid'


def unify_errors(errors):
    label_logger.debug('Unifying errors in %s', errors)
    unique_errors = [list(set(t)) for t in errors]
    label_logger.debug(
        'Unique errors in %s are: %s',
        errors,
        unique_errors
    )
    max_score_errors = [
        {
            error: functools.reduce(
                max,
                [
                    _couple[1]
                    for _couple
                    in filter(
                        lambda couple: couple[0] == error,
                        _unique_errors
                    )
                ]
            )
            for error in dict(_unique_errors).keys()
        }
        for _unique_errors in unique_errors
    ]
    label_logger.debug(
        'Maximum score errors in %s are: %s',
        errors,
        max_score_errors
    )
    return [lint(error) for error in max_score_errors]


def class_error(errors):
    label_logger.debug('Meta class %s score', errors)
    return 0 if errors == 'Valid' else max(errors.values())


def errors(error_dict, hierarchical=True, depth=2, LoD=2, threshold=5):
    label_logger.info(
        'Extracting %s hierarchical errors in %s with depth %s, LoD %s and '
        + 'threshold %s',
        '' if hierarchical else 'non',
        error_dict,
        depth,
        LoD,
        threshold
    )
    label_logger.info('Getting errors in array shape...')
    unq_array, bul_array, fac_array = error_arrays(error_dict, threshold)
    label_logger.debug(
        'Error arrays are %s: ',
        [unq_array, bul_array, fac_array]
    )
    unq, bul, fac = [
        int(sum(meta) > 0) for meta in [unq_array, bul_array, fac_array]
    ]
    label_logger.debug(
        'Array compiling presence of errors %s',
        [unq, bul, fac]
    )
    if depth == 0:
        return 'Unqualifiable' if unq > 0 else 'Qualifiable'
    elif depth == 1:
        return 'Unqualifiable' if unq > 0 else (
            'Error' if bul + fac > 0 else 'Valid'
        )
    elif depth == 2:
        if hierarchical:
            return CLASSES[
                unq * 1
                +
                int(LoD > 0) * (1 - unq) * bul * 2
                +
                int(LoD > 1) * (1 - unq) * (1 - bul) * fac * 3
            ]
        else:
            return 'Unqualifiable' if unq > 0 else (
                int(LoD > 0) * [bul] + int(LoD > 1) * [fac]
            )
    elif depth == 3:
        if hierarchical:
            if unq > 0:
                return 'Unqualifiable'
            elif unq + bul + fac == 0:
                return ('Valid', None)
            else:
                return (
                    (
                        ('Building', bul_array) if LoD > 0 else ('Valid', None)
                    ) if bul else fac * (
                        ('Facet', fac_array) if LoD > 1 else ('Valid', None)
                    )
                )
        else:
            return 'Unqualifiable' if unq > 0 else (
                (LoD > 0) * bul_array
                +
                (LoD > 1) * fac_array
            )
    else:
        raise ValueError


def exists(sub_error, error_type, errors, threshold):
    label_logger.info(
        'Existance of sub_error %s of error_type %s in %s with threshold %s',
        sub_error,
        error_type,
        errors,
        threshold
    )
    if errors[ERROR_CATEGORY_INDEX[error_type]] == 'Valid':
        return 0
    elif sub_error not in errors[ERROR_CATEGORY_INDEX[error_type]].keys():
        return 0
    else:
        return int(
            errors[ERROR_CATEGORY_INDEX[error_type]][sub_error] >= threshold
        )


def error_array(error_dict, threshold, error_type):
    label_logger.info(
        'Errors in %s of type %s in array shape with threshold %s',
        error_dict,
        error_type,
        threshold
    )

    return list(
        map(
            lambda sub_error: exists(
                sub_error,
                error_type,
                error_dict,
                threshold
            ),
            ERROR_CATEGORY_DICTIONARY[error_type]
        )
    )


def error_arrays(error_dict, threshold):
    label_logger.info(
        'Errors %s in array shape with threshold %s',
        error_dict,
        threshold
    )
    return [
        error_array(error_dict, threshold, error_type)
        for error_type in ['Unqualifiable', 'Building', 'Facet']
    ]


def get_category_errors(filename, error_category):
    return read(filename)[
        ERROR_CATEGORY_INDEX[error_category]
    ]


def error_couple(filename, first, second):
    return (
        get_category_errors(filename, first),
        get_category_errors(filename, second)
    )


def similtaneous_errors_list(labels_dir, files, first, second):
    return filter(
        lambda couple: couple[0] not in ERROR_DICTIONARY['Valid']
        and couple[1] not in ERROR_DICTIONARY['Valid'],
        [
            error_couple(os.path.join(labels_dir, f), first, second)
            for f in files
        ]
    )


def similtaneous_errors_lists():
    pass


def similtaneous_errors_scores(labels_dir, files, first, second):
    return [
        (max(dic_couple[0].values()), max(dic_couple[1].values()))
        for dic_couple
        in similtaneous_errors_list(labels_dir, files, first, second)
    ]


def score_similtaneous_errors(labels_dir, files, first, second):
    return float(
        functools.reduce(
            lambda x, y: x + y,
            [
                min(couple)
                for couple
                in similtaneous_errors_scores(labels_dir, files, first, second)
            ]
        )
    ) / 10.


def print_similtaneous_summary(labels_dir, files, first, second):
    joint = score_similtaneous_errors(labels_dir, files, first, second)
    print(
        second + ' errors and ' + first + ' errors joint probability: ',
        joint / float(len(files)),
        '\n' + second + ' errors probability knowing ' + first + ' errors: ',
        joint / score_category_errors(first, labels_dir, files),
        '\nBuilding errors  knowing no Facet errors',

    )


def list_category_errors(error_category, labels_dir, files):
    return filter(
        lambda _error: _error not in ERROR_DICTIONARY['Valid'],
        [
            get_category_errors(os.path.join(labels_dir, f), error_category)
            for f
            in files
        ]
    )


def score_category_errors(error_category, labels_dir, files):
    return float(
        functools.reduce(
            operator.add,
            [
                max(dic.values())
                for dic in list_category_errors(
                    error_category,
                    labels_dir,
                    files
                )
            ]
        )
    ) / 10.


def errors_statistics(error, error_category, labels_dir, files):
    category_errors = list_category_errors(error_category, labels_dir, files)
    number_of_category_errors = score_category_errors(
        error_category,
        labels_dir,
        files
    )
    errors = filter(
        lambda _errors: functools.reduce(
            lambda x, y: x or y,
            [error in _errors.keys()]
        ),
        category_errors
    )
    number_of_errors = float(
        functools.reduce(
            operator.add,
            [max(dic.values()) for dic in errors]
        )
    ) / 10.

    return (
        number_of_errors / number_of_category_errors,
        number_of_errors / float(len(files))
    )


def print_statistics(error, error_category, labels_dir, files):
    x, y = errors_statistics(error, error_category, labels_dir, files)
    print(
        'ratio of \'',
        error,
        '\' error :\n   - among \'',
        error_category,
        '\' errors: ',
        x,
        '\n   - among all files: ',
        y
    )


def print_statistics_summary(error_category, labels_dir, files):
    print(
        'ratio of \'',
        error_category,
        '\' errors among all errors:',
        score_category_errors(error_category, labels_dir, files)
        /
        float(len(files))
    )
    map(
        lambda error: print_statistics(
            error,
            error_category,
            labels_dir,
            files
        ),
        ERROR_CATEGORY_DICTIONARY[error_category]
    )


def summarize_statistics(error_category, labels_dir, files):
    print(error_category, ':', ERROR_CATEGORY_DICTIONARY[error_category])
    print(
        error_category,
        ':',
        map(
            lambda error: errors_statistics(
                error,
                error_category,
                labels_dir,
                files
            ),
            ERROR_CATEGORY_DICTIONARY[error_category]
        )
    )


def main():
    labels_dir = os.path.join(
        '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/',
        'export-3DS/_labels'
    )
    files = fnmatch.filter(os.listdir(labels_dir), '*.shp')

    error_dict = read(os.path.join(labels_dir, files[0]))
    print(error_dict)
    print(error_arrays(error_dict, 5))
    print(LABELS(2, ['Building']))

    # write(
    #     '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/export-3DS'
    #     + '/_ground_truth.csv',
    #     {
    #         os.path.splitext(_file)[0]:
    #         read(
    #             os.path.join(labels_dir, _file)
    #         )
    #         for _file in files
    #     }
    # )

    print(
        {
            os.path.splitext(_file)[0]:
            read(
                os.path.join(labels_dir, _file)
            )
            for _file in files
        }
    )

    print(
        labels_map(
            '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/export-3DS'
            + '/ground_truth.csv',
            True,
            3,
            2,
            5,
            filetype='csv'
        )
    )


if __name__ == '__main__':
    main()

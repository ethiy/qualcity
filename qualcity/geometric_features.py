# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import operator
import functools

from tqdm import tqdm

import numpy as np

import networkx as nx

from . import utils

geom_logger = logging.getLogger(__name__)

NODE_ATTRIBUTES = ['degree', 'area', 'circumference']
EDGE_ATTRIBUTES = ['centroid', 'angle']


def read_features(line):
    geom_logger.debug('Reading face features from line %s...', line)
    (
        face_id,
        degree,
        area,
        circumference,
        centroid0,
        centroid1,
        centroid2,
        normal0,
        normal1,
        normal2
    ) = line.split(" ")
    return (
        int(face_id),
        (
            int(degree),
            float(area),
            float(circumference),
            np.array(
                [
                    float(centroid0),
                    float(centroid1),
                    float(centroid2)
                ]
            ),
            np.array(
                [
                    float(normal0),
                    float(normal1),
                    float(normal2)
                ]
            )
        )
    )


def get_lines(filename):
    geom_logger.info('Lines of %s', filename)
    geom_logger.info('Opening file %s in read mode', filename)
    with open(filename, 'r') as _file:
        lines = list(_file)
        geom_logger.debug('Lines are \n: %s', lines)
    return [line.split('\n')[0] for line in lines if line != '\n']


def get_faces(lines):
    geom_logger.debug('Faces and their attributes in %s...', lines)
    return [read_features(face) for face in lines[:len(lines) // 2]]


def get_adjacency_matrix(lines):
    geom_logger.debug('Facets adjacency matrix in %s', lines)
    return np.array(
        [
            [int(bit) for bit in line.split(" ")]
            for line in lines[len(lines) // 2:-1]
        ]
    )


def get_graph(lines):
    geom_logger.debug('Graph from %s', lines)
    return nx.from_numpy_matrix(
        get_adjacency_matrix(
            lines
        )
    )


def get_relations(faces, adjacency_matrix):
    geom_logger.debug('Pairs of related faces in %s from %s', faces, adjacency_matrix)
    return [
        (faces[i][0], faces[j][0])
        for i, j in zip(
            *np.where(adjacency_matrix == 1)
        )
        if i != j
    ]


def node_statistics(faces, attribute, functions, **kwargs):
    geom_logger.info('Facet %s statistics: %s', attribute, functions)
    return utils.stats(
        [
            face[NODE_ATTRIBUTES.index(attribute)]
            for face in faces.values()
        ],
        functions
    )


def edge_statistics(faces, attribute, functions, relations=[], **kwargs):
    geom_logger.info(
        'Facets %s statistics (i.e. %s) with%s relations',
        attribute,
        functions,
        'out' if len(relations) == 0 else ''
    )
    if len(relations) == 0:
        relations = [
            (i, j)
            for i in faces.keys()
            for j in faces.keys()
            if i != j
        ]
    return utils.stats(
        [
            np.linalg.norm(
                faces[idx][
                    len(NODE_ATTRIBUTES) + EDGE_ATTRIBUTES.index(attribute)
                ]
                -
                faces[_idx][
                    len(NODE_ATTRIBUTES) + EDGE_ATTRIBUTES.index(attribute)
                ]
            )
            for idx, _idx in relations
        ],
        functions
    )


def attribute_statistics(lines, geom_attrib, functions, **kwargs):
    geom_logger.info(
        'Getting %s statistics for attribute %s',
        functions,
        geom_attrib
    )
    geom_logger.info('Getting facets...')
    faces = get_faces(lines)
    adjacency_matrix = get_adjacency_matrix(lines)
    geom_logger.debug('Facets in %s : %s...', lines, faces)
    return {
        'degree':
        lambda faces, functions, **kwargs: node_statistics(
            dict(faces),
            'degree',
            functions,
            **kwargs
        ),
        'area':
        lambda faces, functions, **kwargs: node_statistics(
            dict(faces),
            'area',
            functions,
            **kwargs
        ),
        'circumference':
        lambda faces, functions, **kwargs: node_statistics(
            dict(faces),
            'circumference',
            functions,
            **kwargs
        ),
        'centroid':
        lambda faces, functions, **kwargs: edge_statistics(
            dict(faces),
            'centroid',
            functions,
            **kwargs
        ),
        'centroid_with_relations':
        lambda faces, functions, **kwargs: edge_statistics(
            dict(faces),
            'centroid',
            functions,
            get_relations(faces, adjacency_matrix),
            **kwargs
        ),
        'angle':
        lambda faces, functions, **kwargs: edge_statistics(
            dict(faces),
            'angle',
            functions,
            **kwargs
        ),
        'angle_with_relations':
        lambda faces, functions, **kwargs: edge_statistics(
            dict(faces),
            'angle',
            functions,
            get_relations(faces, adjacency_matrix),
            **kwargs
        )
    }[geom_attrib](faces, functions, **kwargs)


def statistics_features(filename, attributes, functions, **kwargs):
    geom_logger.info(
        'Getting %s attributes for %s using %s',
        attributes,
        filename,
        functions
    )
    try:
        geom_logger.info('Getting lines in %s...', filename)
        lines = get_lines(filename)
        geom_logger.info('Finished getting lines in %s...', filename)
        return functools.reduce(
            lambda _list, attr: _list + attribute_statistics(
                lines,
                attr,
                functions,
                **kwargs
            ),
            attributes,
            [len(lines) // 2]
        )
    except Exception:
        geom_logger.exception('Could not extract features for %s:', filename)


def get_method(graph_dir, method, **parameters):
    if method == 'statistics':
        return lambda building: statistics_features(
            os.path.join(graph_dir, building + '.txt'),
            **parameters
        )
    elif method == 'histogram':
        return lambda building: statistics_features(
            os.path.join(graph_dir, building + '.txt'),
            parameters['attributes'],
            method,
            **parameters['parameters']
        )
    else:
        raise NotImplementedError(
            '{} is not implemented'.format(method)
        )


def geometric_features(buildings, graph_dir, **kwargs):
    geom_logger.info(
        'Getting geometric features for buildings %s in %s using %s with parameters %s',
        buildings,
        graph_dir,
        kwargs['method'],
        kwargs['parameters']
    )
    return {
        building: np.array(
            get_method(
                graph_dir,
                kwargs['method'],
                **kwargs['parameters']
            )(building)
        )
        for building in tqdm(
            buildings,
            desc='Geometric features'
        )
    }


def read(filename):
    geom_logger.info('Read %s and construct corresponding graph.', filename)
    geom_logger.info('Getting lines in %s...', filename)
    lines = get_lines(filename)
    geom_logger.info('Finished getting lines in %s...', filename)

    geom_logger.info('Getting facets...')
    faces = dict(get_faces(lines))
    geom_logger.debug('Facets in %s : %s...', lines, faces)
    geom_logger.info('Getting the graph structure...')
    G = get_graph(lines)
    geom_logger.debug('The graph structure in %s : %s...', lines, G)
    nx.set_node_attributes(
        G,
        {idx: faces[idx][0] for idx in range(len(faces))},
        'degree'
    )
    nx.set_node_attributes(
        G,
        {idx: faces[idx][1] for idx in range(len(faces))},
        'area'
    )
    nx.set_node_attributes(
        G,
        {idx: faces[idx][2] for idx in range(len(faces))},
        'centroid'
    )
    nx.set_node_attributes(
        G,
        {idx: faces[idx][3] for idx in range(len(faces))},
        'normal'
    )
    return G

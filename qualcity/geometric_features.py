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


def node_statistics(faces, attribute, statistics, **kwargs):
    geom_logger.info('Facet %s statistics: %s', attribute, statistics)
    return utils.stats(
        [
            face[NODE_ATTRIBUTES.index(attribute)]
            for face in faces.values()
        ],
        statistics
    )


def edge_statistics(faces, attribute, statistics, relations=[], **kwargs):
    geom_logger.info(
        'Facets %s statistics (i.e. %s) with%s relations',
        attribute,
        statistics,
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
        statistics
    )


def attribute_statistics(lines, geom_attrib, statistics, **kwargs):
    geom_logger.info(
        'Getting %s statistics for attribute %s',
        statistics,
        geom_attrib
    )
    geom_logger.info('Getting facets...')
    faces = get_faces(lines)
    adjacency_matrix = get_adjacency_matrix(lines)
    geom_logger.debug('Facets in %s : %s...', lines, faces)
    return {
        'degree':
        lambda faces, statistics, **kwargs: node_statistics(
            dict(faces),
            'degree',
            statistics,
            **kwargs
        ),
        'area':
        lambda faces, statistics, **kwargs: node_statistics(
            dict(faces),
            'area',
            statistics,
            **kwargs
        ),
        'circumference':
        lambda faces, statistics, **kwargs: node_statistics(
            dict(faces),
            'circumference',
            statistics,
            **kwargs
        ),
        'centroid':
        lambda faces, statistics, **kwargs: edge_statistics(
            dict(faces),
            'centroid',
            statistics,
            **kwargs
        ),
        'centroid_with_relations':
        lambda faces, statistics, **kwargs: edge_statistics(
            dict(faces),
            'centroid',
            statistics,
            get_relations(faces, adjacency_matrix),
            **kwargs
        ),
        'angle':
        lambda faces, statistics, **kwargs: edge_statistics(
            dict(faces),
            'angle',
            statistics,
            **kwargs
        ),
        'angle_with_relations':
        lambda faces, statistics, **kwargs: edge_statistics(
            dict(faces),
            'angle',
            statistics,
            get_relations(faces, adjacency_matrix),
            **kwargs
        )
    }[geom_attrib](faces, statistics, **kwargs)


def features(filename, attributes, statistics, **kwargs):
    geom_logger.info(
        'Getting %s attributes for %s using %s',
        attributes,
        filename,
        statistics
    )
    try:
        geom_logger.info('Getting lines in %s...', filename)
        lines = get_lines(filename)
        geom_logger.info('Finished getting lines in %s...', filename)
        return functools.reduce(
            lambda _list, attr: _list + attribute_statistics(
                lines,
                attr,
                statistics,
                **kwargs
            ),
            attributes,
            [len(lines) // 2]
        )
    except Exception:
        geom_logger.exception('Could not extract features for %s:', filename)


def geometric_features(buildings, graph_dir, attributes, statistics, **parameters):
    geom_logger.info(
        'Getting geometric features for buildings %s in %s based %s and %s',
        buildings,
        graph_dir,
        attributes,
        statistics
    )
    return {
        graph: np.array(
            features(
                os.path.join(graph_dir, graph + '.txt'),
                attributes,
                statistics,
                **parameters
            )
        )
        for graph in tqdm(
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

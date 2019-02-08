# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import operator
import functools

from tqdm import tqdm

import numpy as np

import networkx as nx
import grakel

from . import utils

geom_logger = logging.getLogger(__name__)

NODE_ATTRIBUTES = ['degree', 'area', 'circumference']
EDGE_ATTRIBUTES = ['centroid', 'normal']


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
            for line in lines[len(lines) // 2:]
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
        'normal':
        lambda faces, functions, **kwargs: edge_statistics(
            dict(faces),
            'normal',
            functions,
            **kwargs
        ),
        'normal_with_relations':
        lambda faces, functions, **kwargs: edge_statistics(
            dict(faces),
            'normal',
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
        return np.array(
            functools.reduce(
                lambda _list, attr: _list + attribute_statistics(
                    lines,
                    attr,
                    functions,
                    **kwargs
                ),
                attributes,
                [len(lines) // 2]
            )
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
    elif method == 'graph':
        return lambda building: graph_features(
            os.path.join(graph_dir, building + '.txt'),
            **parameters
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
        kwargs['parameters'] if 'parameters' in kwargs.keys() else {}
    )
    return {
        building: get_method(
            graph_dir,
            kwargs['method'],
            **kwargs['parameters'] if 'parameters' in kwargs.keys() else {}
        )(building)
        for building in tqdm(
            buildings,
            desc='Geometric features'
        )
    }


def get_edges(lines):
    geom_logger.info('Read pairs from lines %s.', lines)
    geom_logger.info('Computing adjacency matrix...')
    adjacency_matrix = get_adjacency_matrix(lines)
    return [
        (i, j)
        for i, j in zip(
            *np.where(adjacency_matrix == 1)
        )
        if i != j
    ]


def get_node_attributes(faces, attributes):
    return {
        idx: [
            faces[face_id][
                NODE_ATTRIBUTES.index(attribute)
            ]
            for attribute in attributes
            if attribute in NODE_ATTRIBUTES
        ] + sum(
            [
                list(
                    faces[face_id][
                        len(NODE_ATTRIBUTES) + EDGE_ATTRIBUTES.index(attribute)
                    ]
                )
                for attribute in attributes
                if attribute in EDGE_ATTRIBUTES
            ],
            list()
        ) if (len(attributes) > 1 or not attributes) else (
            faces[face_id][
                NODE_ATTRIBUTES.index(attributes[0])
            ] if attributes[0] in NODE_ATTRIBUTES else faces[face_id][
                len(NODE_ATTRIBUTES) + EDGE_ATTRIBUTES.index(attributes[0])
            ]
        )
        
        for (idx, face_id)
        in enumerate(faces)
    }


def get_edge_attributes(edges, faces, attributes):
    face_indexing = dict(enumerate(faces))
    return {
        edge: [
            np.linalg.norm(
                faces[face_indexing[edge[0]]][
                    len(NODE_ATTRIBUTES) + EDGE_ATTRIBUTES.index(attribute)
                ]
                -
                faces[face_indexing[edge[1]]][
                    len(NODE_ATTRIBUTES) + EDGE_ATTRIBUTES.index(attribute)
                ]
            )
            for attribute in attributes
        ] if len(attributes) > 1 else np.linalg.norm(
            faces[face_indexing[edge[0]]][
                len(NODE_ATTRIBUTES) + EDGE_ATTRIBUTES.index(attributes[0])
            ]
            -
            faces[face_indexing[edge[1]]][
                len(NODE_ATTRIBUTES) + EDGE_ATTRIBUTES.index(attributes[0])
            ]
        )
        for edge in edges
    }


def graph_features(filename, **parameters):
    if list(parameters.keys()) == ['list']:
        return {
            graph_format:
            extract_graph(filename, **format_parameters)
            for graph_format, format_parameters in parameters['list'].items()
        }
    else:
        raise NotImplementedError(
            '{} is not implemented'.format(parameters)
        )


def extract_graph(filename, node_attributes=[], edge_attributes=[]):
    geom_logger.info('Read %s and construct corresponding graph.', filename)
    geom_logger.info('Getting lines in %s...', filename)
    lines = get_lines(filename)
    geom_logger.info('Getting edges...')
    edges = get_edges(lines)
    if not node_attributes and not edge_attributes:
        return [edges]
    else:
        geom_logger.info('Getting facets...')
        faces = dict(get_faces(lines))
        geom_logger.debug('Facets in %s : %s...', lines, faces)
        geom_logger.info('Extracting node attributes.')
        node_features = get_node_attributes(faces, node_attributes)
        if not edge_attributes:
            return (edges, node_features)
        else:
            geom_logger.info('Extracting edge attributes.')
            return (
                edges,
                node_features,
                get_edge_attributes(edges, faces, edge_attributes)
            )


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
        {idx: faces[face_id][0] for (idx, face_id) in enumerate(faces)},
        'degree'
    )
    nx.set_node_attributes(
        G,
        {face_id: faces[face_id][1] for (idx, face_id) in enumerate(faces)},
        'area'
    )
    nx.set_node_attributes(
        G,
        {face_id: faces[face_id][2] for (idx, face_id) in enumerate(faces)},
        'circumference'
    )
    nx.set_node_attributes(
        G,
        {face_id: faces[face_id][3] for (idx, face_id) in enumerate(faces)},
        'centroid'
    )
    nx.set_node_attributes(
        G,
        {face_id: faces[face_id][4] for (idx, face_id) in enumerate(faces)},
        'normal'
    )
    return G

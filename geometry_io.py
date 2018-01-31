#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import operator

import functools

import numpy as np

import networkx as nx

import matplotlib.pyplot as plt

import utils

geom_logger = logging.getLogger('qualcity.' + __name__)


def read_features(line):
    geom_logger.debug('Reading face features from line %s...', line)
    (
        face_id,
        degree,
        area,
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
    return lines


def get_faces(lines):
    geom_logger.debug('Faces and their attributes in %s...', lines)
    return dict([read_features(face) for face in lines[:len(lines) // 2]])


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


def get_relations(lines):
    geom_logger.debug('Pairs of related faces in %s', lines)
    return [
        (i, j)
        for i, j in zip(
            *np.where(get_adjacency_matrix(lines) == 1)
        )
        if i != j
    ]


def degree_statistics(faces, statistics, **kwargs):
    geom_logger.info('Facet degree statistics (i.e. %s)', statistics)
    return utils.stats([face[0] for face in faces.values()], statistics)


def area_statistics(faces, statistics, **kwargs):
    geom_logger.info('Facet area statistics (i.e. %s)', statistics)
    return utils.stats([face[1] for face in faces.values()], statistics)


def centroid_statistics(faces, statistics, relations=[], **kwargs):
    geom_logger.info(
        'Facets centroid statistics (i.e. %s) with%s relations',
        statistics,
        'out' if len(relations) == 0 else ''
    )
    if len(relations) == 0:
        relations = [
            (idx, _idx)
            for idx in faces.keys()
            for _idx in faces.keys()
            if idx != _idx
        ]
    return utils.stats(
        [
            np.linalg.norm(faces[idx][2] - faces[_idx][2])
            for idx, _idx in relations
        ],
        statistics
    )


def angle_statistics(faces, statistics, relations=[], **kwargs):
    geom_logger.info(
        'Facets angle statistics (i.e. %s) with%s relations',
        statistics,
        'out' if len(relations) == 0 else ''
    )
    if len(relations) == 0:
        relations = [
            (idx, _idx)
            for idx in faces.keys()
            for _idx in faces.keys()
            if idx != _idx
        ]
    return utils.stats(
        [
            np.arctan2(
                np.linalg.norm(np.cross(faces[idx][3], faces[_idx][3])),
                np.dot(faces[idx][3], faces[_idx][3])
            ) * 180 / np.pi
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
    geom_logger.debug('Facets in %s : %s...')
    return {
        'degree': degree_statistics,
        'area': area_statistics,
        'centroid': centroid_statistics,
        'centroid_bis':
        lambda faces, statistics, **kwargs: centroid_statistics(
            faces,
            statistics,
            get_relations(lines),
            **kwargs
        ),
        'angle': angle_statistics,
        'angle_bis': lambda faces, statistics, **kwargs: angle_statistics(
            faces,
            statistics,
            get_relations(lines),
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


def geometric_features(graph_dir, attributes, statistics, **parameters):
    geom_logger.info(
        'Getting geometric features for all files in %s based %s and %s',
        graph_dir,
        attributes,
        statistics
    )
    return {
        os.path.splitext(graph)[0]: np.array(
            features(
                os.path.join(graph_dir, graph),
                attributes,
                statistics,
                **parameters
            )
        )
        for graph in fnmatch.filter(
            os.listdir(graph_dir),
            '*.txt'
        )
    }


def read(filename):
    geom_logger.info('Read %s and construct corresponding graph.', filename)
    geom_logger.info('Getting lines in %s...', filename)
    lines = get_lines(filename)
    geom_logger.info('Finished getting lines in %s...', filename)

    geom_logger.info('Getting facets...')
    faces = get_faces(lines)
    geom_logger.debug('Facets in %s : %s...', lines, faces)
    geom_logger.info('Getting the graph structure...')
    G = get_graph(lines)
    geom_logger.debug('The graph structure in %s : %s...', lines, G)
    nx.set_node_attributes(
        G, 'degree', {idx: faces[idx][0] for idx in range(len(faces))})
    nx.set_node_attributes(
        G, 'area', {idx: faces[idx][1] for idx in range(len(faces))})
    nx.set_node_attributes(
        G, 'centroid', {idx: faces[idx][2] for idx in range(len(faces))})
    nx.set_node_attributes(
        G, 'normal', {idx: faces[idx][3] for idx in range(len(faces))})
    return G


def main():
    graph_dir = os.path.join(
        '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/export-3DS',
        'dual_graphs'
    )
    print(
        read(os.path.join(graph_dir, '3078.txt')).node[1]
    )
    print(
        features(
            os.path.join(graph_dir, '3078.txt'),
            ['degree', 'area'],
            ['min', 'max']
        )
    )

    print(
        features(
            os.path.join(graph_dir, '3078.txt'),
            ['degree', 'area'],
            'histogram',
            bins=10
        )
    )

    nx.draw(read(os.path.join(graph_dir, '3078.txt')))
    plt.show()


if __name__ == '__main__':
    main()

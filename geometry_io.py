#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import operator

import numpy as np

import networkx as nx

import matplotlib.pyplot as plt

import utils


def read_features(line):
    logging.info('Read face features...')
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
    logging.info('Getting lines of %s', filename)
    with open(filename, 'r') as file:
        lines = list(file)
    return lines


def get_faces(filename):
    logging.info('Getting faces and their attributes in %s...', filename)
    lines = get_lines(filename)
    return dict([read_features(face) for face in lines[:len(lines) / 2]])


def get_adjacency_matrix(filename):
    logging.info('Getting the facets adjacency matrix in %s', filename)
    lines = get_lines(filename)
    return np.array(
        [
            [int(bit) for bit in line.split(" ")]
            for line in lines[len(lines) / 2:-1]
        ]
    )


def get_graph(filename):
    logging.info('Getting the graph in %s', filename)
    return nx.from_numpy_matrix(
        get_adjacency_matrix(
            filename
        )
    )


def get_relations(filename):
    logging.info('Getting pairs of related faces in %s', filename)
    i, j = np.where(get_adjacency_matrix(filename) == 1)
    return filter(lambda x: x[0] != x[1], zip(*[i, j]))


def degree_statistics(faces, statistics, **kwargs):
    logging.info('Getting facet degree statistics (i.e. %s)', statistics)
    return utils.stats([face[0] for face in faces.values()], statistics)


def area_statistics(faces, statistics, **kwargs):
    logging.info('Getting facet area statistics (i.e. %s)', statistics)
    return utils.stats([face[1] for face in faces.values()], statistics)


def centroid_statistics(faces, statistics, relations=[], **kwargs):
    logging.info(
        'Getting facets centroid statistics (i.e. %s) with%s relations',
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
    logging.info(
        'Getting facets angle statistics (i.e. %s) with%s relations',
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


def attribute_statistics(filename, geom_attrib, statistics, **kwargs):
    logging.info(
        'Getting %s statistics for attribute %s in %s',
        statistics,
        geom_attrib,
        filename
    )
    facets = get_faces(filename)
    return {
        'degree': degree_statistics,
        'area': area_statistics,
        'centroid': centroid_statistics,
        'centroid_bis': lambda faces, statistics: centroid_statistics(
            faces,
            statistics,
            get_relations(filename)
        ),
        'angle': angle_statistics,
        'angle_bis': lambda faces, statistics: angle_statistics(
            faces,
            statistics,
            get_relations(filename)
        )
    }[geom_attrib](facets, statistics, **kwargs)


def features(filename, attributes, statistics, **kwargs):
    logging.info(
        'Getting %s attributes for %s using %s',
        attributes,
        filename,
        statistics
    )
    return reduce(
        lambda _list, attr: _list + attribute_statistics(
            filename,
            attr,
            statistics,
            **kwargs
        ),
        attributes,
        [len(get_faces(filename))]
    )


def geometric_features(graph_dir, attributes, statistics, **kwargs):
    logging.info(
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
                statistics
            )
        )
        for graph in fnmatch.filter(
            os.listdir(graph_dir),
            '*.txt'
        )
    }


def read(filename):
    logging.info('Read %s and construct corresponding graph.', filename)
    faces = get_faces(filename)
    G = get_graph(filename)
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
    print read(os.path.join(graph_dir, '3078.txt')).node[1]
    print features(
        os.path.join(graph_dir, '3078.txt'),
        ['degree', 'area'],
        ['min', 'max']
    )

    print features(
        os.path.join(graph_dir, '3078.txt'),
        ['degree', 'area'],
        'histogram',
        bins=10
    )

    nx.draw(read(os.path.join(graph_dir, '3078.txt')))
    plt.show()


if __name__ == '__main__':
    main()

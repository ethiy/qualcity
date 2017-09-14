#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils import median, mean


def read_features(line):
    face_id, degree, area, centroid0, centroid1, centroid2, normal0, normal1, normal2 = line.split(
        " ")
    return (
        int(face_id),
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


def get_lines(filename):
    with open(filename, 'r') as file:
        lines = list(file)
    return lines


def get_faces(filename):
    lines = get_lines(filename)
    return [read_features(face) for face in lines[:len(lines) / 2]]


def get_adjacency_matrix(filename):
    lines = get_lines(filename)
    return np.array(
        [
            [int(bit) for bit in line.split(" ")]
            for line in lines[len(lines) / 2:-1]
        ]
    )


def get_graph(filename):
    return nx.from_numpy_matrix(
        get_adjacency_matrix(
            filename
        )
    )


def degree_statistics(faces):
    degrees = [face[1] for face in faces]
    return [min(degrees), max(degrees), mean(degrees), median(degrees)]


def area_statistics(faces):
    areas = [face[2] for face in faces]
    return [min(areas), max(areas), mean(areas), median(areas)]


def feature_vector(filename):
    faces = get_faces(filename)
    return [len(faces)] + degree_statistics(faces) + area_statistics(faces)


def read(filename):
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
    root_path = os.path.join(
        '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/export-3DS',
        'dual_graphs'
    )
    print read(os.path.join(root_path, '3078.txt')).node[1]
    print feature_vector(os.path.join(root_path, '3078.txt'))

    nx.draw(read(os.path.join(root_path, '3078.txt')))
    plt.show()


if __name__ == '__main__':
    main()

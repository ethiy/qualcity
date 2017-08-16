#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def read_features(line):
    degree, area, centroid0, centroid1, centroid2, normal0, normal1, normal2 = line.split(" ")
    return (int(degree), float(area), np.array([float(centroid0), float(centroid1), float(centroid2)]), np.array([float(normal0), float(normal1), float(normal2)]))

def read_graph(lines):
    return nx.from_numpy_matrix(np.array([[int(bit) for bit in line .split(" ")] for line in lines]))

def read(filename):
    with open(filename, 'r') as file:
        lines = list(file)
        faces = [read_features(face) for face in lines[:len(lines)/2]]
        G = read_graph(lines[len(lines)/2:-1])
        nx.set_node_attributes(G, 'degree', {idx : faces[idx][0] for idx in range(len(faces))})
        nx.set_node_attributes(G, 'area', {idx : faces[idx][1] for idx in range(len(faces))})
        nx.set_node_attributes(G, 'centroid', {idx : faces[idx][2] for idx in range(len(faces))})
        nx.set_node_attributes(G, 'normal', {idx : faces[idx][3] for idx in range(len(faces))})
        return G

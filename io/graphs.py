#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import os
import numpy as np
import networkx as nx

def read_features(line):
    degree, area, centroid0, centroid1, centroid2, normal0, normal1, normal2 = line.split(" ")
    return (int(degree), float(area), np.array([float(centroid0), float(centroid1), float(centroid2)]), np.array([float(normal0), float(normal1), float(normal2)]))

def read_graph(lines):
    return nx.from_numpy_matrix(np.array([[int(bit) for bit in line .split(" ")] for line in lines]))

def read(filename):
    with open(filename, 'r') as file:
        lines = list(file)
        faces = [read_features(face) for face in lines[:len(lines)/2]]
        return (faces, read_graph(lines[len(lines)/2:-1]).edges())

def main():
    root_path = '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/export-3DS/dual_graphs'
    print read(os.path.join(root_path, '3078.txt'))

if __name__ == '__main__':
    main()

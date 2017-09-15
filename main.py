#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import os
import fnmatch

import graph_io
import labels_io


def graph_files(directory):
    return fnmatch.filter(
        [os.path.join(directory, file) for file in os.listdir(directory)],
        '*.txt'
    )


def main():
    root_path = os.path.join(
        '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/export-3DS',
        'dual_graphs'
    )

    graphs = [
        graph_io.read(graph_file) for graph_file in graph_files(root_path)
    ]


if __name__ == '__main__':
    main()

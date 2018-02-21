#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

"""preprocess.

Usage:
    preprocess.py (-h | --help)
    preprocess.py prune <projection> --dual-graph <dual_graph>

Options:
    -h --help           Show this screen.

"""

import docopt

import geometry_io


def main():
    arguments = docopt.docopt(
        __doc__,
        help=True,
        version=None,
        options_first=False
    )

    print(arguments)


if __name__ == '__main__':
    main()

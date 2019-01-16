#!/usr/bin/env python3.5
# -*- coding: <utf-8> -*-

import numpy as np
import matplotlib.pyplot as plt


def read(filename):
    with open(filename, 'r') as fi:
        lines = fi.readlines()
        return (lines[0], lines[1:5]+lines[6:])


def prune(feature_importances, sizes=[0, 10, 20, 40]):
    weighted = np.array(
        [
            np.sum(feature_importances[first:last]) / (last - first)
            for (first, last) in zip(
                np.cumsum(sizes[:-1]),
                np.cumsum(sizes[1:])
            )
        ]
    )
    return weighted / np.sum(weighted)


def extract(line):
    return np.array(
        [
            float(n) 
            for n in line.split('\n')[0].split(': ')[1][1:-1].split()
        ]
    )


def main():
    feature_order, fis_lines = read('paris_2_fi.txt')
    features = feature_order.split()
    fis = [
        prune(extract(line), sizes=[0, 20, 20, 20])
        for line in fis_lines
    ]

    titles = [
        'BOS',
        'BUS',
        'BIB',
        'BIT',
        'FOS',
        'FUS',
        'FIT',
        'FIB',
        'FIG'
    ]

    bul, axes = plt.subplots(1, 4)
    # bul.set_size_inches(4*10, 4)
    for (ax, fi, title) in zip(axes, fis[:4], titles[:4]):
        ax.pie(
            fi,
            labels=features,
            autopct='%1.1f%%',
            shadow=True,
            startangle=90
        )
        ax.axis('square')
        ax.set_title(title)
    bul.show()

    fac, axes = plt.subplots(1, 5)
    # fac.set_size_inches(5*12, 4)
    for (ax, fi, title) in zip(axes, fis[4:], titles[4:]):
        ax.pie(
            fi,
            labels=features,
            autopct='%1.1f%%',
            shadow=True,
            startangle=90
        )
        ax.axis('square')
        ax.set_title(title)
    fac.show()

    plt.show()


if __name__ == "__main__":
    main()

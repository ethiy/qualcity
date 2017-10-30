#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import operator
import pdb


def median(iterable):
    length = len(iterable)
    sorted_iterable = sorted(iterable)
    if length == 0:
        raise LookupError
    elif length % 2 == 1:
        return sorted_iterable.__getitem__(length // 2)
    else:
        return (
            sorted_iterable.__getitem__(length // 2 - 1) +
            sorted_iterable.__getitem__(length // 2)
        ) / 2.


def mean(iterable):
    return sum(iterable) / len(iterable)


def fuse_elements(elements):
    s, elements = zip(
        *sorted(
            enumerate(elements),
            key=lambda c: len(c[1])
        )
    )

    keys, values = zip(
        *map(
            lambda el: zip(*el),
            elements
        )
    )

    if not set(keys[0]) <= set(keys[1]):
        raise LookupError

    values_2 = [val for key, val in elements[1] if key in set(keys[0])]

    return list(
            zip(
                keys[0],
                [
                    (
                        couple[s[0]],
                        couple[s[1]]
                    )
                    for couple
                    in zip(
                        values[0],
                        values_2
                    )
                ]
            )
        ) + [
            (key, (None, val))
            if s[0] == 0
            else (key, (val, None))
            for key, val in elements[1]
            if key not in set(keys[0])
        ]


def fuse(dict_1, dict_2):
    return dict(
        fuse_elements(
            map(
                lambda d: sorted(d, key=operator.itemgetter(0)),
                [dict_1.items(), dict_2.items()]
            )
        )
    )


def main():
    print(median([-5, -5, -3, -4, 0, -1]))
    print(median((-5, -5, -3, -4, 0, -1)))
    print(mean((-5, -5, -3, -4, 0, -1)))
    print(median([-5, -5, -4, 0, -1]))
    print(mean([-5, -5, -4, 0, -1]))

    print(
        fuse(
            {'4': 545, '10': 641, '8': 45},
            {'4': 15, '10': 8574}
        )
    )


if __name__ == '__main__':
    main()

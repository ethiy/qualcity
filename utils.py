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


def fuse_elements(elements_1, elements_2):
    elements_1, elements_2 = sorted(
        [elements_1, elements_2],
        key=len
    )

    (keys_1, values_1), (keys_2, _) = map(
        lambda el: zip(*el),
        (elements_1.items(), elements_2.items())
    )

    if not set(keys_1) <= set(keys_2):
        raise LookupError

    values_2 = [val for key, val in elements_2.items() if key in set(keys_1)]
    print(values_2)
    return list(
            zip(
                keys_1,
                zip(
                    values_1,
                    values_2
                )
            )
        ) + [
            (key, val)
            for key, val in elements_2.items()
            if key not in set(keys_1)
        ]


def fuse(dict_1, dict_2):
    return fuse_elements(dict_1, dict_2)


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

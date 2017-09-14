#! /usr/bin/env python
# -*- coding: <utf-8> -*-


def median(iterable):
    length = len(iterable)
    if length == 0:
        raise LookupError
    elif length % 2 == 1:
        return sorted(iterable).__getitem__(length // 2)
    else:
        return (
            sorted(iterable).__getitem__(length // 2 - 1) +
            sorted(iterable).__getitem__(length // 2)
        ) / 2.


def mean(iterable):
    return sum(iterable) / len(iterable)


def main():
    print median([-5, -5, -3, -4, 0, -1])
    print median((-5, -5, -3, -4, 0, -1))
    print mean((-5, -5, -3, -4, 0, -1))
    print median([-5, -5, -4, 0, -1])
    print mean([-5, -5, -4, 0, -1])


if __name__ == '__main__':
    main()

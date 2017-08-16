#! /usr/bin/env python
# -*- coding: <utf-8> -*-

from setuptools import setup

def readme():
    """
    Longer description from readme.
    """
    with open('ReadMe.md', 'r') as readmefile:
        return readmefile.read()

setup(name = 'qualcity',
    version = '0.0.0',
    description = 'Prototyping building qualification.',
    long_description=readme(),
    classifiers=[
        'License :: MIT License',
        'Programming Language :: Python :: 2.7'
    ],
    keywords='qualification building 3d reconstruction graphs computer vision',
    url='https://gitlab.com/Ethiy/qualicity',
    author='Oussama Ennafii',
    author_email='oussama.ennafii@ign.fr',
    license='MIT',
    packages=[
        'qualcity',
        'qualcity.io',
        'qualcity.graphs',
        'qualcity.features',
        'qualcity.features.kernels'
    ],
    install_requires=[
        'networkx',
        'numpy',
        'matplotlib',
        'scipy'
    ],
    include_package_data=True,
    zip_safe=False)

#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

from setuptools import setup, find_packages


def readme():
    """
    Longer description from readme.
    """
    with open('ReadMe.md', 'r') as readmefile:
        return readmefile.read()


def requirements():
    """
    Get requirements to install.
    """
    with open('requirements.txt', 'r') as requirefile:
        return [line.strip() for line in requirefile.readlines() if line]


setup(
    name='qualcity',
    version='0.1.0a0',
    description='3D Building model qualification.',
    long_description=readme(),
    classifiers=[
        'License :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Research',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Operating System :: Windows',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.5'
    ],
    scripts=['QualCity'],
    keywords='qualification building 3d reconstruction graphs computer vision',
    url='https://github.com/ethiy/qualcity',
    author='Oussama Ennafii [IGN :: LaSTIG]',
    author_email='oussama.ennafii@ign.fr',
    license='MIT',
    packages=find_packages(exclude=['tests']),
    install_requires=requirements(),
    include_package_data=True,
    zip_safe=False
)

#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

from setuptools import setup, find_packages
from qualcity import config

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
    version=config.__version__,
    description='3D Building model qualification.',
    long_description=readme(),
    classifiers=[
        'License :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    platforms=[
        'Environment :: Console',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Operating System :: Windows'
    ],
    scripts=['qual-city'],
    keywords='qualification building 3d reconstruction graphs computer vision',
    url='https://github.com/ethiy/qualcity',
    author='Oussama Ennafii [IGN :: LaSTIG]',
    author_email='oussama.ennafii@ign.fr',
    license='License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    packages=find_packages(exclude=['tests']),
    install_requires=requirements(),
    include_package_data=True,
    zip_safe=False
)

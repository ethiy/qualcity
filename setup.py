#! /usr/bin/env python
# -*- coding: <utf-8> -*-

def readme():
    """
    Longer description from readme.
    """
    with open('ReadMe.md', 'r') as readmefile:
        return readmefile.read()

setup(name = 'qualify.buildings',
    version = 'dev',
    description = 'Prototyping building qualification.',
    long_description=readme(),
    classifiers=[
        'License :: MIT License',
        'Programming Language :: Python :: 2.7'
    ],
    keywords='qualification building 3d reconstruction graphs computer vision',
    url='https://gitlab.com/Ethiy/prototyping',
    author='Oussama Ennafii',
    author_email='oussama.ennafii@ign.fr',
    license='MIT',
    packages=['qualify.buildings'],
    install_requires=[
        'networkx',
        'numpy',
        'os',
        'matplotlib',
        'scipy'
    ],
    include_package_data=True,
    zip_safe=False)

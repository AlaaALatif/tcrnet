import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='tcrnet',
    version='0.0.1',
    description='Bioinformatics library for the processing and analysis of T-Cell Receptor (TCR) data.',
    author='Alaa Abdel Latif',
    author_email='al.a.latif94@gmail.com',
    url='https://github.com/AlaaALatif/tcrnet',
    packages=find_packages(include=['tcrnet', 'tcrnet.*']),
    install_requires=[
        'pandas>=2.1.4',
        'numpy>=1.24.3',
        'matplotlib>=2.2.0',
        'seaborn>=0.12.2',
        'jupyterlab'
    ],
    extras_require={'plotting': ['jupyter']}
)
import setuptools
import os
import sys

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(here, 'gds'))
from version import __version__

print(f'Version {__version__}')

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gds",
    version=__version__,
    author="GDS team",
    author_email="gds@cs.umd.edu",
    url="https://johnding1996.github.io/Graph-Distribution-Shift/",
    description="GDS graph distribution shift benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    license='MIT',
    packages=setuptools.find_packages(
        exclude=['experiments', 'experiments.models', 'tests', 'pages', 'preprocessing', 'visualization']),
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)

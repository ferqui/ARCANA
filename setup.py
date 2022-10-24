#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst", encoding="utf-8") as readme_file:
    readme = readme_file.read()

__version__ = '0.1.0'

requirements = [
    "torch",
    "numpy",
    "matplotlib",
    "brian2",
    "sympy==1.4",
    "jupyter",
    "jupyterlab"
]

version = __version__

setup(
    author="Fernando M. Quintana",
    author_email="fernando.quintana@uca.es",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    description="DynapSE neural model simulation on PyTorch.",
    install_requires=requirements,
    #license="MIT License",
    long_description=readme,
    include_package_data=True,
    keywords="DynapSE, DynapSEtorch",
    name="DynapSEtorch",
    packages=find_packages(include=["DynapSEtorch", "DynapSEtorch.*"]),
    #test_suite="tests",
    #tests_require=test_requirements,
    url="https://github.com/ferqui/DynapSEtorch",
    version=__version__,
    zip_safe=False,
)
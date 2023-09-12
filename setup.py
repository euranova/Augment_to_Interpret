""" augment_to_interpret is the library used to produce the results of the paper
Augment to Interpret: Unsupervised and Inherently Interpretable Graph Embeddings
"""

import os

from setuptools import setup, find_packages


# Utility function to read the README file for the long description.
def read(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding="utf-8") as file:
        return file.read()


setup(
    name="augment_to_interpret",
    version="0.1",
    author="Anonymous Author",
    author_email="",
    description="Produces the results of the paper Augment to Interpret: "
                "Unsupervised and Inherently Interpretable Graph Embeddings",
    # license="",
    packages=find_packages(include=["augment_to_interpret"]),
    long_description=read("README.md"),
    # install_requires=[],
    # python_requires=[],
)

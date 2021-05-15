#!/usr/bin/env python3

from setuptools import setup, find_packages


with open("README.md") as f:
    README = f.read()

with open("requirements.txt") as f:
    REQUIREMENTS = f.read()


setup(
    name="scalingqa",
    version="1.0.1",
    description="Scaling QA",
    long_description=README,
    #license=LICENCE,
    python_requires=">=3.6",
    packages=find_packages(exclude=("data", "configurations", "experiments", "*.tests", "*.tests.*", "tests.*", "tests")),
    install_requires=REQUIREMENTS.strip().split('\n'),
)

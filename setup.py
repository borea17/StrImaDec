from distutils.core import setup
from setuptools import find_packages

setup(
    name="strimadec",
    version="1.0",
    license="MIT",
    description="Structured Image Decomposition Reference Implementation",
    author="Markus Borea",
    author_email="borea17@protonmail.com",
    url="https://github.com/borea17/strimadec",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
# Author: Taoz
# Date  : 12/7/2018
# Time  : 8:37 AM
# FileName: setup.py
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="esheep-sdk",
    version="0.0.1-alpha1",
    author="seekloud",
    author_email="zta@outlook.com",
    description="sdk for esheep api(https://github.com/seekloud/esheep-api).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seekloud/esheep-sdk",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
    ],
    license='Apache 2.0',
    keywords='reinforcement learning',
)
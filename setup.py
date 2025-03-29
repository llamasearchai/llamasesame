#!/usr/bin/env python3
"""
Setup script for LlamaSesame Voice Cloning Studio Pro
"""

import os
from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read the long description from README.md if it exists
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="llamasesame",
    version="1.0.0",
    description="LlamaSesame Voice Cloning Studio Pro",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LlamaSearch",
    author_email="info@llamasearch.ai",
    url="https://github.com/llamasearch/llamasesame",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    keywords="voice cloning, speech synthesis, ai, deep learning",
    entry_points={
        "console_scripts": [
            "llamasesame=llamasesame.llamasesame:main",
        ],
    },
) 
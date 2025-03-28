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
    name="llamasesame-llamasearch",
    version="1.0.0",
    description="LlamaSesame Voice Cloning Studio Pro",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LlamaSearch AI",
    author_email="nikjois@llamasearch.ai",
    url="https://llamasearch.ai",
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
    package_dir={"": "src"},
    packages=find_packages(where="src"),
) 
# Updated in commit 5 - 2025-04-04 17:18:58

# Updated in commit 13 - 2025-04-04 17:18:58

# Updated in commit 21 - 2025-04-04 17:18:59

# Updated in commit 29 - 2025-04-04 17:18:59

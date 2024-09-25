"""Setuptools installation file for the package."""

from pathlib import Path

from setuptools import find_packages, setup

required = Path("requirements.txt").read_text().splitlines()

setup(
    name="dygnn",
    version="0.1.0",
    packages=find_packages(),
    package_data={"cafein": ["py.typed"]},
    python_requires=">=3.11",
    install_requires=required,
)

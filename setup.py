from setuptools import find_packages
from setuptools import setup

extras = {
    'doc': ['sphinx==5.0.2', 'sphinx-rtd-theme']
}

setup(
    name="cbo",
    version="0.1",
    description="Refactored version of CBO",
    author="",
    author_email="",
    url="https://github.com/ChampiB/CBO_with_OOP",
    license="",
    packages=find_packages(),
    include_package_data=True,
    scripts=[
        "bin/test_graph"
    ],
    install_requires=[
        "numpy~=1.23.5",
        "pandas~=1.5.2",
        "scipy~=1.9.3",
        "matplotlib~=3.6.2",
        "seaborn~=0.12.1",
        "emukit~=0.4.10",
        "GPy~=1.10.0",
        "scikit-learn~=1.2.1",
        "paramz~=0.9.5",
        "hydra-core~=1.3",
        "networkx~=3.0",
    ],
    extras_require=extras,
)

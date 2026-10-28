from setuptools import find_packages, setup

setup(
    name="tide-velocity",
    version="0.1.0",
    description="TIDE: pluggable velocity-graph construction and high-cosine gene module analysis",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "numpy>=1.22",
        "pandas>=1.5",
        "scipy>=1.8",
        "anndata>=0.8",
        "scanpy>=1.9",
        "scvelo>=0.2.5,<0.3",
    ],
)

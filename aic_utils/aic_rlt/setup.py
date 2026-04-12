from setuptools import setup, find_packages

setup(
    name="aic_rlt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "numpy",
    ],
    python_requires=">=3.10",
)

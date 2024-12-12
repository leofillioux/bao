from setuptools import find_packages, setup

setup(
    name="bao",
    packages=find_packages(),
    install_requires=[
        "h5py",
        "pandas",
        "gdown",
        "PyYAML",
        "easydict",
        "scikit-learn",
        "torch",
        "lightning",
        "matplotlib",
        "torchvision",
    ],
)

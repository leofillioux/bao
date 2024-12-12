from setuptools import find_packages, setup

setup(
    name="bao",
    packages=find_packages(),
    install_requires=[
        "h5py",
        "pandas", # can be removed since it is used only once to read a csv file
        "gdown",
        "PyYAML",
        "easydict", # do we really need this? 
        "scikit-learn", # doe we really need this?
        "torch",
        "lightning",
        "matplotlib", # potentially also not necesssary if we want to keep the library lightweight
        "torchvision",
    ],
)

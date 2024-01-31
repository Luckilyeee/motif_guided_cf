from setuptools import setup, find_packages

setup(
    name='naive_shapelet_CF',
    version='0.1.2',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'tslearn',
        'pyts',
        'scikit-learn',
        'matplotlib',
        'scipy',
    ],
    author='Peiyu Li',
    author_email='peiyu.li@usu.edu',
)

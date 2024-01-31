from setuptools import setup, find_packages

setup(
    name='naive-shapelet-CF',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'tslearn',
        'pyts',
        'sklearn',
        'matplotlib',
        'scipy',
    ],
    author='Peiyu Li',
    author_email='peiyu.li@usu.edu',
)

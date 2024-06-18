from setuptools import setup

setup(
    name='borealis-postprocessors',
    version='1.0.0',
    description='Post-process SuperDARN Borealis data files',
    author='Remington Rohel',
    license='GPLv3',
    requires=['numpy', 'scipy', 'pyDARNio', 'h5py', 'deepdish']
)

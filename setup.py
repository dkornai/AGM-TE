from setuptools import setup, find_packages

setup(
    name='agm_te',
    version='0.1',
    packages=find_packages(),
    description='Approximate Generative Model estimation of Transfer Entropy',
    author='Daniel Kornai',
    install_requires=[
        'matplotlib>=3.7.4',
        'numpy>=1.23.5',
        'torch>=2.2.2'
    ]
)
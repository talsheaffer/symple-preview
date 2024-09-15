from setuptools import find_packages, setup

setup(
    name='symple',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/galvinograd/symple',
    description='A symbolic simplifier for mathematical expressions',
    install_requires=[
        'sympy>=1.13',
        'torch>=2.4',
        'tqdm>=4.62',
        'jupyter>=1.0',
        'matplotlib>=3.4',
        'pytest>=8.0',
    ],
)
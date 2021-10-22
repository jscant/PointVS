from distutils.core import setup

setup(
    packages=['point_vs'],
    name='point_vs',
    version='0.1.0',
    description='SE(3)-equivariant neural networks for virtual screening.',
    author='Jack Scantlebury',
    install_requires=[
        'scipy'
    ],
)

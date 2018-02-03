import sys

from setuptools import setup

name = 'nnweaver'
version = '0.1'
release = '0.1'

try:
    import pypandoc

    long_description = pypandoc.convert('README.md', 'rst')
except Exception:
    long_description = open('README.md').read()

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

setup(
    name=name,
    version=version,
    description='A tiny Python library to create and train feedforward neural networks',
    long_description=long_description,
    url='https://github.com/gvinciguerra/nnweaver',
    packages=['nnweaver'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[
        'numpy>=1.14.0',
        'scipy>=1.0.0',
        'tqdm>=4.19.5',
        'matplotlib>=2.1.2'
    ],
    setup_requires=pytest_runner,
    tests_require=['pytest'],
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release)
        }
    }
)

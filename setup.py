import re
import sys

from setuptools import setup

name = 'nnweaver'
version = '0.2'
release = '0.2'


def get_pypi_compatible_description():
    readme = open('README.md').read()
    long_description = re.sub(r'<.*>', '', readme, re.DOTALL)
    long_description = re.sub(r'\[`(.*?)`\]', r'[\1]', long_description)

    try:
        import pypandoc
        long_description = pypandoc.convert_text(long_description, 'rst',
                                                 format='markdown_github')
    except (ImportError, RuntimeError):
        long_description = readme
    return long_description


needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

setup(
    name=name,
    version=version,
    description='A tiny Python library to create and train feedforward neural networks',
    long_description=get_pypi_compatible_description(),
    url='https://github.com/gvinciguerra/nnweaver',
    packages=['nnweaver'],
    python_requires='>=3.5',
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

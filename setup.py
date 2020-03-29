from aughie import __version__

import setuptools


with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='aughie',
    version=__version__,
    author='Agost Biro',
    author_email='agost.biro+aughie_py@gmail.com',
    description='Provides access to the Neural Network Database API',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/aughie/aughie-py',
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas==0.23.4',
        'requests==2.20.1'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ]
)

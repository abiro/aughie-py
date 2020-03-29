.. Aughie Python Client Library documentation master file, created by
   sphinx-quickstart on Thu Sep 20 10:47:42 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Aughie Python Client Library
========================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   api

Overview
--------

The Aughie Python client library provides access to the Neural Network Database API.

The project is currently in a proof-of-concept phase.

Install
-------

Install the library from the Python Package Index with :code:`pip`:

.. code:: bash

    pip install aughie

Aughie needs Python 3.5 or later.

Example
-------

This code snippet returns all networks from the database that have a
:code:`Conv2D` layer with a :code:`tanh` activation and a :code:`MaxPooling2D`
layer and presents the data as a `Pandas <https://pandas.pydata.org/>`_ data
frame.

.. code:: python

    from aughie import nndb

    client = nndb.ApiClient()

    layers = [{'type': 'Conv2D', 'activation': 'tanh'},
              {'type': 'MaxPooling2D'}]
    networks = client.get_networks(layers=layers)

    df = networks.to_data_frame()

Tutorial
--------

The `Neural Network Database API Intro <https://github.com/aughie/aughie-py/blob/master/docs/examples/nndb_api.ipynb>`_
tutorial is a `Jupyter notebook <https://jupyter-notebook.readthedocs.io/en/stable/>`_
that shows how to use the database to analyze the relationship between optimizers and loss
functions and introduces more fine-grained queries such as searching for networks
used for image generation based on layer patterns.

You can also run the tutorial online directly from your browser by following
the link to `Google Colab. <https://colab.research.google.com/github/aughie/aughie-py/blob/master/docs/examples/nndb_api.ipynb>`_


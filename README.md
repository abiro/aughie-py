# Aughie Python Client Library [DEPRECATED]

The Aughie Python client library provides access to the Neural Network Database API.

The project is currently in a proof-of-concept phase.

## Install

Install the library from the Python Package Index with `pip`:

`pip install aughie`

Aughie needs Python 3.5 or later.

## Example

This code snippet returns all networks from the database that have a `Conv2D` layer with a `tanh`
activation and a `MaxPooling2D` layer and presents the data as a [Pandas](https://pandas.pydata.org/) data frame.

```python
from aughie import nndb

client = nndb.ApiClient()
networks = client.get_networks(layers=[{'type': 'Conv2D', 'activation': 'tanh'},
                                       {'type': 'MaxPooling2D'}])
df = networks.to_data_frame()
```

## Tutorial

The [Neural Network Database API Intro](docs/examples/nndb_api.ipynb) tutorial is a [Jupyter notebook](https://jupyter-notebook.readthedocs.io/en/stable/)
that shows how to use the database to analyze the relationship between optimizers and loss 
functions and introduces more fine-grained queries such as searching for networks used for image generation based on 
layer patterns.

You can also run the tutorial online directly from your browser by following 
the link to [Google Colab.](https://colab.research.google.com/github/aughie/aughie-py/blob/master/docs/examples/nndb_api.ipynb) 

## Docs

The documentation is available at: https://aughie-py.readthedocs.io

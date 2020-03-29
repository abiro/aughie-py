import abc
import copy
import functools
import inspect
import json
from typing import Any, Iterable, List, Optional, Tuple, Union

import pandas as pd

import requests

from .cache import Cache


@functools.total_ordering
class TextType(abc.ABC):
    """Provides methods for comparison based on the string representation."""

    @abc.abstractmethod
    def __str__(self) -> str:
        """Get the string representation."""

    def __eq__(self, other: Any) -> bool:
        """Compare equality with an other object."""
        return str(self) == str(other)

    def __lt__(self, other: Any) -> bool:
        """Compare less than with an other object."""
        return str(self) < str(other)

    def __hash__(self) -> int:
        """Get the hash value of this object."""
        return hash(str(self))

    def __repr__(self) -> str:
        """Get the debugging representation."""
        return str(self)


class QueryFields(TextType):
    """Represents GraphQL query fields."""

    def __init__(self, fields: Iterable):
        """Initialize a QueryFields instance.

        Args:
            fields: The field or fields to query. Nested fields must be dicts
                with one key, the name of the dict. Nested fields are
                recursively converted to QueryFields instances.

        Raises:
            ValueError if a field dict doesn't have exactly one key.
            RecursionError if there are too many nested fields.

        """
        self._name = None

        if isinstance(fields, dict):
            if len(fields) != 1:
                raise ValueError('Dict must have exactly one key: {}'.format(
                    fields))
            # Don't use popitem to avoid mutating the input.
            self._name, fields = next(iter(fields.items()))

        if isinstance(fields, str):
            self._fields = [fields]
        else:
            self._fields = list(fields)

        # Recursively convert dicts to QueryFields.
        for i, f in enumerate(self._fields):
            if isinstance(f, dict):
                self._fields[i] = self.__class__(f)

        self._fields.sort()
        fields = ' '.join(map(str, self._fields))

        if self._name:
            self._str = '{name} {{{fields}}}'.format(name=self._name,
                                                     fields=fields)
        else:
            self._str = '{fields}'.format(fields=fields)

    def __str__(self) -> str:
        """Get string representation of a QueryFields instance."""
        return self._str


class Query(TextType):
    """Represents a GraphQL query."""

    def __init__(self,
                 query_template: str,
                 args: dict,
                 fields: Iterable):
        """Initialize a Query instance.

        Args:
            query_template: The query template. Must contain a "fields" named
                format argument. (Eg.: "{fields}").
            args: The query arguments as a dict.
            fields: The requested field or fields.

        Raises:
            Exceptions raised by QueryFields.__init__

        """
        q_fields = QueryFields(fields)
        # Remove unnecessary white space.
        qt = inspect.cleandoc(query_template)
        self._query_str = qt.format(fields=q_fields)

        self._args = args
        self._args_str = json.dumps(self._args, sort_keys=True)

        self._str = json.dumps({'query': self._query_str,
                                'variables': self._args},
                               sort_keys=True)

    def __str__(self):
        """Get the string representation of a Query."""
        return self._str

    def as_params(self) -> List[Tuple[str, str]]:
        """Get the query as url params."""
        return [('query', self._query_str), ('variables', self._args_str)]


class ApiError(Exception):
    """Base class of all api errors."""

    pass


class ApiResult(abc.ABC):
    """Base class for API results with different getters."""

    @abc.abstractmethod
    def to_data_frame(self, fields: Optional[List[str]] = None) -> pd.DataFrame:  # noqa 501
        """Return the API results a Pandas DataFrame instance."""

    @abc.abstractmethod
    def to_list(self) -> list:
        """Return the API results as a list."""


class NetworksResult(ApiResult):
    """Represents data for multiple networks from the API."""

    def __init__(self, networks: List[dict]):
        """Initialize a NetworksResult instance.

        Args:
            networks: The networks as a list of dicts.

        """
        self._networks = networks

    @classmethod
    def _network_to_record(cls, network: dict, parent_key: str = None) -> dict:
        record = {}
        for k, v in network.items():
            if isinstance(v, dict):
                record.update(cls._network_to_record(v, parent_key=k))
            elif v is not None:
                if k == 'type' and parent_key is not None:
                    record[parent_key] = v
                else:
                    record[k] = v
        return record

    @classmethod
    def _network_to_records(cls, network: dict) -> Iterable[dict]:
        if 'losses' in network:
            if network['losses']:
                losses = network['losses']
            else:
                losses = [None]
            del network['losses']
            for loss in losses:
                network['loss'] = loss
                yield cls._network_to_record(network)
        else:
            yield cls._network_to_record(network)

    def to_data_frame(self, fields: Optional[List[str]] = None) -> pd.DataFrame:  # noqa 501
        """Get the networks as a Pandas DataFrame instance.

        Args:
            fields: Optional list with the desired field names. Can be also
                used to set the order of the DataFrame columns.
        Returns:
            The DataFrame instance.
        """
        flat_networks = []
        networks = copy.deepcopy(self._networks)
        for net in networks:
            flat_networks.extend(self._network_to_records(net))
        df = pd.DataFrame(flat_networks)
        if fields is not None:
            return df[fields]
        else:
            return df

    def to_list(self) -> List[dict]:
        """Get the networks as a list of dicts."""
        return copy.deepcopy(self._networks)


class ApiClient:
    """A client to communicate with the Neural Network Database API.

    The client is thread-safe.

    """

    _default_endpoint = 'https://nndb-api.aughie.org/graphql'

    _fields = tuple(['name', 'numLayers', 'numInputs', 'numOutputs',
                             {'optimizer': ['type', 'learningRate',
                                            'hasDecay']},
                             {'losses': 'type'}])
    _cache = Cache()

    @classmethod
    def _normalize_args(cls, args):
        """Sort lists on the first level."""
        args = copy.deepcopy(args)
        res = {}
        for k, v in args.items():
            if v is None:
                continue
            if isinstance(v, list):
                v.sort(key=lambda el: json.dumps(el, sort_keys=True))
            res[k] = v
        return res

    def __init__(self, api_endpoint: str = _default_endpoint,
                 use_cache: bool = True):
        """Initialize an ApiClient instance.

        Args:
            api_endpoint: The API endpoint. Defaults to the official one.
            use_cache: Whether to use a client cache that is shared between
                all ApiClient instances. Defaults to true.
        """
        self._api_endpoint = api_endpoint
        self._use_cache = use_cache

    def _make_query(self, query: Query) -> Any:
        if self._use_cache:
            try:
                return self._cache.get(query)
            except KeyError:
                pass

        try:
            req = requests.get(self._api_endpoint, params=query.as_params())
            res = req.json()
            data = res['data']
            if self._use_cache:
                self._cache.set(query, data)
            return data
        except KeyError:
            raise ApiError(res['errors'])
        except Exception as e:
            raise ApiError(e)

    def get_network(self, network_name: str,
                    fields: Iterable[Union[dict, str]] = _fields) \
            -> NetworksResult:
        """Query a network by its name.

        Args:
            network_name: The name of the network.
            fields: The fields to retrieve from the network. Defaults to all.

        Returns:
            The result as a NetworksResult instance.

        Raises:
            ApiError

        """
        template = """
        query GetNetwork($name: String!) {{
          network(name: $name) {{
            {fields}
          }}
        }}
        """
        q = Query(template, {'name': network_name}, fields)
        res = self._make_query(q)
        return NetworksResult([res['network']])

    def get_networks(self,
                     network: Optional[dict] = None,
                     optimizers: Optional[List[dict]] = None,
                     losses: Optional[List[dict]] = None,
                     layers: Optional[List[dict]] = None,
                     fields: Iterable[Union[dict, str]] = _fields) -> NetworksResult:  # noqa 501
        """Query networks filtered by their optimizers, losses or layers.

        The filters are conjunctive, but the sequences of optimizer and loss
        filters are disjunctive in their categories. The sequence of layer
        filters are conjunctive within their categories as well.
        So the following call:


        .. code-block:: python

            client.get_networks(network={'minNumLayers': 10},
                                optimizers=[{'type': 'Adam'},
                                            {'type': 'SGD'}],
                                losses=[{'type': 'binary_crossentropy'},
                                        {'type': 'categorical_crossentropy'}],
                                layers=[{'type': 'Conv2D',
                                         'activation': 'tanh'},
                                        {'type': 'MaxPooling2D'}])

        will return all networks that have

        * a minimum of 10 layers AND

        * use Adam OR SGD optimizer AND

        * use binary OR categorical cross entropy loss AND

        * have at least one Conv2D layer with a tanh activation AND at least \
        one MaxPooling2D layer.

        See the input objects of the GraphQL schema for a complete list of the
        filter attributes.

        Args:
            network: Network properties to filter by.
            optimizers: Multiple optimizer properties to filter by.
            losses: Multiple loss properties to filter by.
            layers: Multiple layer properties to filter by.
            fields: The fields to retrieve from the network. Defaults to all.

        Returns:
            The results as a NetworksResult instance.

        Raises:
            ApiError

        """
        template = """
        query GetNetworks($network: NetworkInput $optimizers: [OptimizerInput]
                          $losses: [LossInput] $layers: [LayerInput]) {{
          networks(network: $network optimizers: $optimizers losses: $losses
                   layers: $layers) {{
            {fields}
          }}
        }}
        """
        args = self._normalize_args({'network': network,
                                     'optimizers': optimizers,
                                     'losses': losses,
                                     'layers': layers})
        q = Query(template, args, fields)
        res = self._make_query(q)
        return NetworksResult(res['networks'])

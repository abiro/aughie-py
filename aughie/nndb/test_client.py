import copy
import inspect
import json
import math
from itertools import permutations, product
from random import Random
from unittest import TestCase, mock

import numpy as np

from .client import ApiClient, ApiError, NetworksResult, Query, QueryFields, \
    TextType


class TestTextType(TestCase):

    def test_init(self):
        with self.assertRaises(TypeError):
            TextType()

    def test_repr(self):
        class Foo(TextType):
            def __str__(self):
                return "foo"
        f = Foo()
        self.assertEqual(repr(f), "foo")


class TestQueryFields(TestCase):

    def test_init(self):
        with self.assertRaises(ValueError):
            QueryFields({'a': 1, 'b': 2})

    def test_str(self):
        qf1 = QueryFields('bar')
        self.assertEqual(str(qf1), 'bar')

        qf2 = QueryFields([{'foo': 'bar'}, 'fubar'])
        self.assertEqual(str(qf2), 'foo {bar} fubar')

        qf3 = QueryFields(['foo', 'bar'])
        self.assertEqual(str(qf3), 'bar foo')

    def test_eq(self):
        qf1 = QueryFields('bar')
        qf2 = QueryFields('bar')
        self.assertEqual(qf1, qf2)

        fields = ['fubar', 'xyz',
                  {'foo': 'bar'},
                  {'baz': ['baz', {'b': 'a'}]}]
        qfields = [QueryFields(fs) for fs in permutations(fields)]
        for first, second in permutations(qfields, 2):
            self.assertEqual(first, second)
            self.assertEqual(hash(first), hash(second))
            self.assertNotEqual(qf1, second)
            self.assertNotEqual(hash(qf1), hash(second))
            self.assertNotEqual(first, qf2)
            self.assertNotEqual(hash(first), hash(qf2))

    def test_lt(self):
        qf1 = QueryFields('foo')
        qf2 = QueryFields('bar')
        self.assertLess(qf2, qf1)
        self.assertGreater(qf1, qf2)


class TestQuery(TestCase):
    template = """
    query Foo($name: String) {{
        bar(name: $name) {{
            {fields}
        }}
    }}
    """

    def test_str(self):

        exp_query = inspect.cleandoc(self.template.format(fields='fubar'))
        q = Query(self.template, {'name': 'baz'}, 'fubar')
        qs = json.dumps({'query': exp_query,
                         'variables': {'name': 'baz'}},
                        sort_keys=True)
        self.assertEqual(str(q), qs)

        fields2 = ['fubar', 'bar', {'baz': 'a'}]
        variables2 = {'fubar': [{'name': 'foo'}, {'name': 'baz'}],
                      'bar': 1}
        q2 = Query(self.template,
                   variables2,
                   fields2)

        exp_query2 = """
        query Foo($name: String) {
            bar(name: $name) {
                bar baz {a} fubar
            }
        }
        """
        qs2 = json.dumps({'query': inspect.cleandoc(exp_query2),
                          'variables': variables2},
                         sort_keys=True)
        self.assertEqual(str(q2), qs2)

    def test_eq(self):
        # Dicts preserve insertion order from Python 3.6

        q1 = Query(self.template, {'name': 'foo'}, ['bar', 'fubar'])
        q2 = Query(self.template, {'name': 'foo'}, ['fubar', 'bar'])
        self.assertEqual(q1, q2)

        q3 = Query(self.template,
                   {'fubar': {'baz': 3, 'bar': 2},
                    'foo': 1},
                   'bar')
        q4 = Query(self.template,
                   {'foo': 1,
                    'fubar': {'bar': 2, 'baz': 3}},
                   'bar')
        self.assertEqual(q3, q4)
        self.assertNotEqual(q1, q4)

        rand = Random(0)

        key_vals = [('a', 1), ('b', {'e': 2}), ('c', 'foo'), ('xyz', math.pi)]
        qargs = [{'name': dict(args)} for args in permutations(key_vals)]
        rand.shuffle(qargs)

        fields = ['fubar', 'xyz',
                  {'foo': 'bar'},
                  {'baz': ['baz', {'b': 'a'}]}]
        qfields = list(permutations(fields))
        rand.shuffle(qfields)

        prev_query = None
        for args, fields in product(qargs, qfields):
            q = Query('name', args, fields)
            if prev_query:
                self.assertEqual(q, prev_query)
                self.assertEqual(hash(q), hash(prev_query))
            prev_query = q

    def test_lt(self):
        q1 = Query(self.template, {'name': 'foo'}, 'bar')
        q2 = Query(self.template, {'name': 'foo'}, 'fubar')
        self.assertLess(q1, q2)
        self.assertGreater(q2, q1)

    def test_as_params(self):
        args = {'name': 'foo'}
        q = Query(self.template, args, 'bar')
        exp_query = inspect.cleandoc(self.template.format(fields='bar'))
        params = [('query', exp_query),
                  ('variables', json.dumps(args))]
        self.assertListEqual(q.as_params(), params)


class TestNetworksResult(TestCase):

    def setUp(self):
        self._networks = [
            {'losses': [{'type': 'mse'}],
             'name': 'eriklindernoren/Keras-GAN/lsgan',
             'optimizer': {'learningRate': 0.00019999999494757503,
                           'type': 'Adam'}},
            {'losses': [{'type': 'categorical_crossentropy'}],
             'name': 'basveeling/wavenet/wavenet',
             'optimizer': {'learningRate': 0.0010000000474974513,
                           'type': 'SGD'}},
            {'losses': [{'type': 'mae'},
                        {'type': 'mae'},
                        {'type': 'mae'},
                        {'type': 'mse'},
                        {'type': 'mse'},
                        {'type': 'mae'}],
             'name': 'eriklindernoren/Keras-GAN/discogan',
             'optimizer': {'learningRate': 0.00019999999494757503,
                           'type': 'Adam'}},
            {'losses': [],
             'name': 'keras-team/keras-applications/inception_resnet_v2',
             'optimizer': None}
        ]

    def test_to_list(self):
        nr = NetworksResult(self._networks)
        list_res = nr.to_list()
        self.assertListEqual(list_res, self._networks)
        list_res[0]['name'] = 'foo'
        list_res_2 = nr.to_list()
        # Test deep copy
        self.assertNotEqual(list_res_2[0]['name'], 'foo')

    def test_to_data_frame(self):
        nets = copy.deepcopy(self._networks)
        nr = NetworksResult(self._networks)
        fields = ['name', 'loss', 'optimizer', 'learningRate']
        df = nr.to_data_frame(fields=fields)
        # Test deep copy
        self.assertListEqual(nr.to_list(), nets)
        # Test columns
        self.assertListEqual(list(df), fields)
        # Test contents
        self.assertEqual(df.shape, (9, 4))
        self.assertEqual(df.iloc[1]['name'], self._networks[1]['name'])
        self.assertEqual(df.iloc[1]['optimizer'],
                         self._networks[1]['optimizer']['type'])
        self.assertEqual(df.iloc[2]['name'], self._networks[2]['name'])
        self.assertEqual(df.iloc[3]['name'], self._networks[2]['name'])
        self.assertEqual(df.iloc[5]['name'], self._networks[2]['name'])
        self.assertEqual(df.iloc[5]['loss'], 'mse')
        self.assertTrue(np.isnan(df.iloc[-1]['optimizer']))
        self.assertTrue(np.isnan(df.iloc[-1]['learningRate']))

        nets = [{'name': 'foo'}, {'name': 'bar'}]
        nr2 = NetworksResult(nets)
        df2 = nr2.to_data_frame()
        self.assertEqual(df2.iloc[1]['name'], 'bar')


class TestApiClient(TestCase):

    def test_endpoint(self):
        # better double check
        self.assertEqual(ApiClient._default_endpoint,
                         'https://nndb-api.aughie.org/graphql')

    def test_normalize_args(self):
        args = {'foo': [{'c': 3}, {'a': 1}, {'b': 2}],
                'bar': {'d': 4},
                'baz': None}
        res = ApiClient._normalize_args(args)
        # Make sure deep copy is used.
        args['bar']['d'] = 5
        exp_res = {'bar': {'d': 4},
                   'foo': [{'a': 1}, {'b': 2}, {'c': 3}]}
        self.assertDictEqual(res, exp_res)

    @mock.patch('aughie.nndb.client.ApiClient._cache')
    @mock.patch('aughie.nndb.client.requests')
    def test_make_query_valid(self, requests_mock, cache_mock):
        cm = cache_mock
        rm = requests_mock
        rm.get.return_value = rm.Request
        rm.Request.json.return_value = {'data': 'query-res'}

        endpoint = 'api-endpoint'
        client = ApiClient(api_endpoint=endpoint)
        q = Query('{fields}', {'arg': True}, 'field')

        cm.get.side_effect = KeyError(str(q))

        res = client._make_query(q)

        rm.get.assert_called_once_with(endpoint,
                                       params=q.as_params())

        cm.get.assert_called_once_with(q)
        cm.set.assert_called_once_with(q, 'query-res')

        self.assertEqual(res, 'query-res')

    @mock.patch('aughie.nndb.client.ApiClient._cache')
    @mock.patch('aughie.nndb.client.requests')
    def test_make_query_no_cache(self, requests_mock, cache_mock):
        cm = cache_mock
        rm = requests_mock
        rm.get.return_value = rm.Request
        rm.Request.json.return_value = {'data': 'query-res'}

        client = ApiClient(use_cache=False)
        q = Query('{fields}', {'arg': True}, 'field')

        res = client._make_query(q)

        rm.get.assert_called_once_with(ApiClient._default_endpoint,
                                       params=q.as_params())

        cm.get.assert_not_called()
        cm.set.assert_not_called()

        self.assertEqual(res, 'query-res')

    @mock.patch('aughie.nndb.client.requests')
    def test_make_query_graphql_error(self, requests_mock):
        r = requests_mock
        r.get.return_value = r.Request
        r.Request.json.return_value = {'errors': ["GraphQL error"]}
        client = ApiClient()
        q = Query('{fields}', {'arg': True}, 'field')
        with self.assertRaises(ApiError, msg='["GraphQL error"]'):
            client._make_query(q)

    @mock.patch('aughie.nndb.client.requests')
    def test_make_query_connection_error(self, requests_mock):
        r = requests_mock
        msg = 'Connection error'
        r.get.side_effect = ConnectionError(msg)
        client = ApiClient()
        q = Query('{fields}', {'arg': True}, 'field')
        with self.assertRaises(ApiError) as cm:
            client._make_query(q)
        self.assertEqual(str(cm.exception), msg)

    @mock.patch('aughie.nndb.client.ApiClient._make_query')
    def test_get_network(self, make_query_mock):
        template = """
        query GetNetwork($name: String!) {{
          network(name: $name) {{
            {fields}
          }}
        }}
        """
        ret_val = {'name': 'network-name', 'losses': ['loss-type']}
        make_query_mock.return_value = {'network': ret_val}
        fields = ['name', {'losses': ['type']}]
        q = Query(template,
                  {'name': 'network-name'},
                  fields)
        client = ApiClient()
        res = client.get_network('network-name', fields=fields)
        make_query_mock.assert_called_once_with(q)
        self.assertListEqual(res.to_list(), [ret_val])

    @mock.patch('aughie.nndb.client.ApiClient._make_query')
    def test_get_networks(self, make_query_mock):
        template = """
        query GetNetworks($network: NetworkInput $optimizers: [OptimizerInput]
                          $losses: [LossInput] $layers: [LayerInput]) {{
          networks(network: $network optimizers: $optimizers losses: $losses
                   layers: $layers) {{
            {fields}
          }}
        }}
        """

        ret_val = [{'name': 'foo', 'optimizer': ['optimizer-type']}]
        make_query_mock.return_value = {'networks': ret_val}

        optimizers_arg = [{'hasDecay': True}]
        losses_arg = [{'type': 'mse'}, {'type': 'binary_crossentropy'}]
        layers_arg = [{'activation': 'relu'}]

        fields = ['name', {'optimizer': ['type']}]
        client = ApiClient()
        res = client.get_networks(optimizers=optimizers_arg,
                                  losses=losses_arg,
                                  layers=layers_arg,
                                  fields=fields)

        losses_arg_sorted = sorted(losses_arg,
                                   key=lambda el: json.dumps(el,
                                                             sort_keys=True))
        q = Query(template,
                  {'optimizers': optimizers_arg,
                   'losses': losses_arg_sorted,
                   'layers': layers_arg},
                  fields)

        make_query_mock.assert_called_once_with(q)
        self.assertListEqual(res.to_list(), ret_val)

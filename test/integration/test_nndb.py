#!/usr/bin/env python3

import sys

from aughie import nndb

if len(sys.argv) == 2:
    api_endpoint = sys.argv[1]
    print('Using endpoint:', api_endpoint)
    client = nndb.ApiClient(api_endpoint=api_endpoint, use_cache=False)
else:
    print('Using default endpoint')
    client = nndb.ApiClient(use_cache=False)

print('Testing get_network')
res1 = client.get_network('zhixuhao/unet/unet').to_list()[0]
assert res1['name'] == 'zhixuhao/unet/unet', res1

print('Testing get_networks')
res2 = client.get_networks(network={'minNumLayers': 10},
                           optimizers=[{'type': 'Adam'},
                                       {'type': 'SGD'}],
                           losses=[{'type': 'binary_crossentropy'},
                                   {'type': 'categorical_crossentropy'}],
                           layers=[{'type': 'Conv2D', 'activation': 'tanh'},
                                   {'type': 'MaxPooling2D'}]).to_list()
assert len(res2) >= 1, res2
for r in res2:
    assert r['numLayers'] >= 10, r

print('All tests OK')

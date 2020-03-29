#!/usr/bin/env bash

set -e

# Force to not exit with error if directory doesn't exist
rm -rf integration_venv
python3.6 -m venv integration_venv
source integration_venv/bin/activate
pip install ../..

# TODO start NNDB api server automatically

python test_nndb.py $1

rm -rf integration_venv

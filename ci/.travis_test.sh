#!/usr/bin/env bash

set -e

python --version
python -c "import torch; print('pytorch %s' % torch.__version__)"


if [[ "$COVERAGE" == "true" ]]; then
     PYTHONPATH='.' pytest -sv --cov=pytorch_coral --doctest-modules coral_pytorch
else
     PYTHONPATH='.' pytest -sv --doctest-modules coral_pytorch
fi

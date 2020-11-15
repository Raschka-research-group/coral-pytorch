#!/usr/bin/env bash

set -e

if [[ "$TRAVIS_OS_NAME" != "osx" ]]; then 
    if [ "${TRAVIS_PYTHON_VERSION}" == "2.7" ]; then

        wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
    else
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    fi
else
    if [ "${TRAVIS_PYTHON_VERSION}" == "2.7" ]; then

        wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh -O miniconda.sh
    else
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
    fi
fi
    

bash miniconda.sh -b -p "$HOME/miniconda"
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda update -q pip
conda info -a

conda create -q -n test-environment python="$MINICONDA_PYTHON_VERSION"
source activate test-environment

if [ "${LATEST}" = "true" ]; then
    conda install pytorch torchvision torchaudio -c pytorch
else
    conda install pytorch="$PYTORCH_VERSION" torchvision torchaudio -c pytorch
fi

conda install pytest

if [ "${COVERAGE}" = "true" ]; then
    conda install pytest-cov
    conda install coveralls
fi

python --version
python -c "import torch; print('pytorch %s' % torch.__version__)"

#!/bin/bash

export PATH=`pwd`/bin:$PATH
export PYTHONPATH=`pwd`/:`pwd`/develop:$PYTHONPATH

if [ ! -d "develop" ]; then
    mkdir develop
    python setup.py develop --install-dir develop
fi


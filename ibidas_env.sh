#!/bin/bash

export PATH=`pwd`/bin:$PATH
export PYTHONPATH=`pwd`/:`pwd`/develop:$PYTHONPATH

if [ ! -d "develop" ]; then
    echo "MKDIR"
    mkdir develop
fi

if [ ! "$(ls -A develop)" ]; then
    bin/python_2_please setup.py develop --install-dir develop
fi


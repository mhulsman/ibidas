Installation
============

If one has the ``setuptools`` package installed, one can simply perform::

    easy_install ibidas

This will download and install ibidas and its necessary dependencies. 
The ``setuptools`` package is available in nearly all Linux distributions. 

Ibidas has currently only been tested on the Linux platform, and probably will 
not work out of the box on Windows. We will look into this in the near future.


Starting ibidas
---------------
To start ibidas, one can simply execute::

    ibidas

on the command line prompt. This will load the ``IPython`` interpreter, with
Ibidas preloaded. 

If one instead wants to load ibidas within an script or interpreter process, 
one can simply use::

    from ibidas import *

For the rest of this tutorial, we assume the standard Ibidas interpreter is used.


Manual installation
-------------------
Alternatively, one can download the source package, and execute in the unpacked
source directory::

    python setup.py install

One can also directly execute ibidas from the source package by performing::

    python setup.py develop

    source set_env.sh

    ibidas

Performing unit tests and building the documentation can be done manually using::

    #unit tests
    python setup.py test

    #build documentation (available then under docs/_build/html)
    python setup.py build_sphinx

Ibidas installation dependencies include:

 * python >= 2.6

 * numpy >= v1.5.1

 * ipython >= 0.10.1

 * sqlalchemy >= 0.6.4

 * sphinx >= 1.0.5

Also, installing database client libraries such as `sqlite`, `postgres` and `mysql`,
will alow one to use the sql wrapper to connect to these various databases. 



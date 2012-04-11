Installation and Use
====================

If one has the ``setuptools`` package installed, one can simply perform::

    easy_install ibidas

One needs to have root access to do this (see `Execute from source` if this is not available).
This will download and install ibidas and its necessary dependencies.
The ``setuptools`` package is available in nearly all Linux distributions. 

To install, Ibidas needs some dependencies. Most are automatically installed.
The following packages need to be installed manually (i.e. using the package manager of your distribution) if they are not 
yet available (an error will be returned in this case):

    * numpy (development) files

    * psycopg2 (only if access is required to postgres databases)

    * MySQLDB (only if access is required to mysql databases)

    * other databases are also supported. An error will be returned once you try to connect, with the package that needs to be installed.
    

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


Execute from source
-------------------
One can also directly execute ibidas from the source package. Download the source from pypi,
then execute::

    tar -xzvf ibidas-0.1.13.tar.gz 

to extract the source (version can differ). Enter the source directory using::

    cd ibidas-0.1.13

then start ibidas using::

    ./run

This will install possible dependencies locally. 

By executing in the source directory::

    source ibidas_env.sh

one will add ibidas to the python path, enabling its use in other scripts using::

    from ibidas import *

Manual installation
-------------------
Alternatively, one can download the source package, and execute in the unpacked
source directory::

    python setup.py install


Performing unit tests and building the documentation can be done manually using::

    #unit tests
    python setup.py test

    #build documentation (available then under docs/_build/html)
    python setup.py build_sphinx

Ibidas dependencies include:

 * python >= 2.6

 * numpy >= v1.4.1

 * ipython >= 0.10.1

 * sqlalchemy >= 0.6.4

 * sphinx >= 1.0.5 [only to build documentation]

Also, installing database client libraries such as `sqlite`, `postgres` and `mysql`,
will alow one to use the sql wrapper to connect to these various databases. 



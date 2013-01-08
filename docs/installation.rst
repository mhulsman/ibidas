Installation and Use
====================

To install, Ibidas needs some dependencies. Most are automatically installed.

The following packages need to be installed manually (i.e. using the package manager of your distribution) if they are not 
yet available (an error will be returned in this case):

    * python (version 2.x, not 3.x!. Development files should also be installed, e.g. something named 'python27-dev' or similar in your package manager)

    * a compiler (e.g. gcc). Available on most/all linux distributions.  

    * psycopg2 (only if access is required to postgres databases)

    * MySQLDB (only if access is required to mysql databases)

    * other databases are also supported. An error will be returned once you try to connect, with the package that needs to be installed.

Next, if one has the ``setuptools`` package installed, one can simply perform::

    sudo easy_install -U ibidas

This will download and install ibidas and some required dependencies. Root/administrator access is required to do this (see `Execute from source` or `Install in a virtualenv` if this is not available). 

.. note::
   Some distributions might have multiple versions of python installed (both a 3.x and a 2.x version), requiring one to specify that the python 2.x version has to be used, by e.g. using 'easy_install-2.7' 
   instead of 'easy_install' (use tab-completion on the command line to find the available easy_install versions).

.. note::
   If the ``setuptools`` package is not available, it is most likely available in the package manager of your distribution. In the unlikely case 
   that it cannot be found there, one can install it as follows::

       wget http://peak.telecommunity.com/dist/ez_setup.py
       sudo python ez_setup.py

   For more documentation, see http://pypi.python.org/pypi/setuptools

.. warning::
   Ibidas is currently only tested on the Linux platform, and probably will not work out of the box on Windows. Installation on Mac is probably possible (using e.g. macports to install numpy/setuptools, then using easy_install to install ibidas), but is untested. 
   We will look into this in the near future.

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
One can also directly execute ibidas from the source package. Note that this still requires the aforementioned dependencies to be installed.

Download the source from pypi, then execute::

    tar -xzvf ibidas-0.1.22.tar.gz 

to extract the source (version number can differ). Enter the source directory using::

    cd ibidas-0.1.22

then start ibidas using::

    ./run

This will install possible dependencies locally. 

By executing in the source directory::

    source ibidas_env.sh

one will add ibidas to the python path, enabling its use in other scripts using::

    from ibidas import *


Install in a virtualenv
-----------------------

Virtualenv allows one to install a python environment in a home directory, removing the need for administrator access. To create a virtual environment, one can use::

    curl -O https://raw.github.com/pypa/virtualenv/master/virtualenv.py

    python virtualenv.py ibidas_env

    chmod +x ibidas_env/bin/activate

(If 'python' is a python 3.x version, again search for a python 2.x version, named e.g. 'python2' (use tab-completion to find the available options)). 

Next, one needs to activate the environment (this has to be done for every terminal that is opened in which the virtualenv is used)::

    ibidas_env/bin/activate

Subsequently, from such a terminal, one can install and start ibidas::

    ibidas_env/bin/pip install ibidas

    ibidas


.. warning::
    We encountered some errors in installing the sqlalchemy dependency. This could simply be solved by installing the dependency manually, before installing ibidas::

       ibidas_env/bin/pip install sqlalchemy


Manual installation
-------------------
Alternatively, one can download the source package, and execute in the unpacked source directory::

    python setup.py install

This requires that any dependencies are installed beforehand. 

Ibidas dependencies include:

 * python >= 2.6

 * numpy >= v1.4.1

 * ipython >= 0.10.1

 * sqlalchemy >= 0.6.4

 * sphinx >= 1.0.5 [only to build documentation]

Also, installing Python database client libraries for `sqlite`, `postgres` or `mysql`,
will allow one to use the sql wrapper to connect to these databases. 

Performing unit tests and building the documentation can be done manually using::

    #unit tests
    python setup.py test

    #build documentation (available then under docs/_build/html)
    python setup.py build_sphinx

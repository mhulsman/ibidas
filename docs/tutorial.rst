Tutorial
========

Installing ibidas
-----------------

If one has the ``setuptools`` package installed, one can simply perform::

    easy_install ibidas

to install ibidas and its necessary dependencies. 


Starting ibidas
---------------
To start ibidas, one can simply execute::

    ibidas

on the command line prompt. This will load the ``IPython`` interpreter, with
Ibidas already loaded. 

If one wants to load ibidas within an script or interpreter process instead, 
one can simply use::

    from ibidas import *

For the rest of this tutorial, we assume the standard ibidas interpreter is loaded.

Representation
--------------

In Ibidas one works with data sources by packaging them within a 'representation object'.

Such data sources can be python objects, files, databases, and so on. 
Once they are wihtin a representation object they are all handled similarly.

We start with a simple example, in which we package an integer object::

    >>> r = rep(3)
    Slices: data  
    Types:  int32
    Dims:         
    
    Data: 3

When one executes ``rep`` without specifying a type, ibidas will automatically 
detect the type itself. In this case, the detected type is ``int32``. 

.. note::
    Depending on the platform you use, the type can also be ``int64``.

.. tip::
    The concepts ``slices`` and ``dims`` will be explained later.

The type determines how operations on the representor are handled.
As this is an integer type, one can perform standard integer operations on the representor::
    
    >>> (r + 3) * r
    Slices: data 
    Types:  int32 
    Dims:         

    Data: 18

Now, lets do this on a list of integers::
    
    >>> r = rep([1,2,3])
    Slices: data
    Types:  int32
    Dims:   d1:3

    Data: [1 2 3]

    >>> (r + 3) * r
    Slices: data  
    Types:  int32 
    Dims:   d1:3  
    
    Data: [ 4 10 18]

One might have noted that, although we now represent a list of integers, the type has not changed.

This is because ``rep`` (by default) **unpacks** the data. By unpacking, operations
will not be performed at the *list* level, but instead at the *list elements* level.

Summary:
    * A representor object encapsulates a data source. 

    * Data sources can be python objects, but also files or databases.

    * The current type describes at which level operations are executed. 


Slices
------





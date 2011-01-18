Tutorial
========

In Ibidas one works with data sources by packaging them within a 'representation object'.

Such data sources can be python objects, files, databases, and so on. 
Once they are wihtin a representation object they are all handled similarly.

We start with a simple example, in which we package a python integer::

    >>> r = rep(3)
    Slices: data  
    Types:  uint8 
    Dims:         
    
    Data: 3


The concepts slice, types and dims will be explained later on.

We can simply apply standard operations to the representor, such as::
    
    >>> r + 3
    Slices: data 
    Types:  uint8 
    Dims:         

    Data: 6

    >>> r * r
    Slices: data 
    Types:  uint8 
    Dims:         

    Data: 9




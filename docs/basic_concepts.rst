Basic concepts
==============

In this section we describe some of the basic underlying concepts of Ibidas.

Representation
--------------

In Ibidas one works with data sources by packaging them within a 'representation object'.

Such data sources can be python objects, files, databases, and so on. 
Once they are wihtin a representation object they are all handled similarly.

We start with a simple example, in which we package an integer object::

    >>> Rep(3)
    Slices: | data 
    ---------------
    Type:   | int64
    Dims:   |      
    Data:   |      
            | 3  


.. tip::
    The concepts ``slices``, ``types`` and ``dims`` will be explained in the next sections


One can perform all kind of operations with data that is in a representation object, e.g::

    >>> Rep([1,2,3]) + 3
    Slices: | data 
    ---------------
    Type:   | int64
    Dims:   | d1:3 
    Data:   |      
            | 4    
            | 5    
            | 6    


Summary:
    * A representor object encapsulates a data source. 

    * Data sources can be python objects, but also files or databases.

Query execution
---------------

You might have noted that executing the previous commands resulted immediatly in a printout 
of the contents of the representation object. This is due to the IPython interpreter, 
which will print a representation of the result of all non-assignment operations. 
So, if we instead would have executed::
    
    >>> r = Rep([1,2,3]) + 3

no output would have been printed. More importantly however, Ibidas would also not have performed the
requested addition operation. I.e. ``r`` would have only been a representation of the requested operations 
and data sources, and thus not the result of those operations.

.. important::
    Operations are executed lazily. I.e. only when output is requested by the user.

The reason for this behaviour is that it allows optimizations which are otherwise not possible. For example, it 
enables the system to translate queries against a database (partially) into the query language SQL. This way, 
instead of moving unnecessary data to Ibidas, operations can be moved to the database.

.. note::
   For brevity, we will sometimes print output in this tutorial after assignments (e.g. ``r = Rep(3)``.
   In reality this does not happen. One can still get the output after such statements, by simply executing ``r``

Often, one wants to just see a description of the contents of a representation object, not the actual data result itself.
This can be done using the information attribute ``I``::

    >>> r = Rep(["geneA","geneB"])
    >>> r.I
    Slices: data     
    Types:  bytes[5] 
    Dims:   d1:2 


Note that the data is not printed. Especially in case of slow operations or data sources this can be useful.

On the other hand, there are also situations in which one wants to force the query to execute, e.g. because
its result is used multiple times (which would otherwise result in a part of the query being executed multiple times). 
Execution can be forced by using the ``Copy()`` command::

    >>> r = Rep([1,2,3]) + 4
    >>> r = r.Copy()


Getting data out of a representor object is simple, one simply appends ``()`` to a query to let it return the 
results as normal ``numpy`` or python objects::

    >>> r()
    array(['geneA', 'geneB'], dtype=object)

As you can see, Ibidas has packaged the data in a numpy array. 

Summary:
    * Operations are only executed when needed, to allow for optimizations

    * One can ask for a description of the representor contents using the ``I`` attribute.

    * Use the ``Copy`` command to force execution of a query.

    * One can get the data results by transforming the query into a function call by appending ``()``

Types
-----
When one executes ``Rep`` without specifying a type, the type is detected automatically.
For example, in the first example, the detect type was ``int64``. 

.. note::
    Depending on the platform you use, the type can also be ``int32``.

The type determines how operations on the representor are handled.
For example, with an integer type, one can perform standard integer operations on the representor::
    
    >>> r = Rep(3)
    >>> (r + 3) * r
    Slices: | data 
    ---------------
    Type:   | int64
    Dims:   |      
    Data:   |      
            | 18


Similarly, in case of the string type, the addition operation becomes concatenation::

    >>> Rep(["geneA", "geneB"])  + "_geneC"
    Slices: | data       
    ---------------------
    Type:   | bytes[11]  
    Dims:   | d1:2       
    Data:   |            
            | geneA_geneC
            | geneB_geneC


One might have noted that, although we now represent a list of thins, the type still represents the
type of the list elements. 

This is because ``Rep`` (by default) **unpacks** the data. By unpacking, operations
will not be performed at the *list* level, but instead at the *list elements* level. Unpacking/packing will be explained
further in one of the next sections.

Summary:
    * A type is assigned automatically when packaging data using ``rep``

    * The type indicates at which data nesting level operations are executed. 


Slices
------

Whereas lists in Ibidas are used to denote collections of data with the same type,
tuples are used to describe data compositions of different types. 

You might have know such compositions as *records*, or simply as table rows.

So, lets load a simple table::
    
    >>> data = [('gene1',0.5), ('gene2', 0.3), ('gene3', 0.8)]
    
    >>> r = Rep(data)
    Slices: | f0       | f1
    ---------------------------
    Type:   | bytes[5] | real64
    Dims:   | d1:3     | d1:3
    Data:   |          |
            | gene1    | 0.5
            | gene2    | 0.3
            | gene3    | 0.8


If we compare the output to earlier output, we now see that there are more than one columns in the data description.

These columns represented *slices*. Slices are one of the main concepts in Ibidas. They can be compared to columns/fields in a table, but are more general.

.. note::
    Note that we use tuples to describe records, and a list to store the records of the table. In Ibidas, tuples are used to relate attributes that describe some
    common object, while lists are used to gather elements with the same type / role. 

Selecting a slice can be done using simple attribute lookup::

    >>> r.f0
    Slices: | f0      
    ------------------
    Type:   | bytes[5]
    Dims:   | d1:3    
    Data:   |         
            | gene1   
            | gene2   
            | gene3  


Each slice has a name (the first row), a type (second row) and a dimension (third row). Dimensions will be explained later. For now, it is important that each
slice has a common type. This means that all data elements adressed by it can be handled in the same way. Slices could thus also be seen as a kind of cursor in your data structure.
Performing operations on this cursor will perform the operations on a subset of your data. For example::

    >>> r.f0 == "gene2"
    Slices: | f0   
    ---------------
    Type:   | bool 
    Dims:   | d1:3 
    Data:   |      
            | False
            | True 
            | False

To select multiple slices, one can use the :py:meth:`ibidas.representor.Representor.get` function::

    >>> r.Get("f1", "f0")
    Slices: | f1     | f0      
    ---------------------------
    Type:   | real64 | bytes[5]
    Dims:   | d1:3   | d1:3    
    Data:   |        |         
            | 0.5    | gene1   
            | 0.3    | gene2   
            | 0.8    | gene3  

    >>> r.Get(1, 0)
    Slices: | f1     | f0      
    ---------------------------
    Type:   | real64 | bytes[5]
    Dims:   | d1:3   | d1:3    
    

    >>> r.Get(r.f1, r.f0)
    Slices: | f1     | f0      
    ---------------------------
    Type:   | real64 | bytes[5]
    Dims:   | d1:3   | d1:3    
    

    >>> r.Get(_.f1, _.f0)
    Slices: | f1     | f0      
    ---------------------------
    Type:   | real64 | bytes[5]
    Dims:   | d1:3   | d1:3    

The last option shows the use of the context operator ``_``, which adresses the enclosing data representor (in this case `r`). This allows one
to refer easily to slices in longer commands. One can also combine this with other operations::

    >>> r.Get(_.f1 + 3, _.f0)
    Slices: | f1     | f0      
    ---------------------------
    Type:   | real64 | bytes[5]
    Dims:   | d1:3   | d1:3    
    Data:   |        |         
            | 3.5    | gene1   
            | 3.3    | gene2   
            | 3.8    | gene3  

One can also use this function to combine slices, e.g::

    >>> r.Get(_.f0, Rep("cancer_genes"))
    Slices: | f0       | data        
    ---------------------------------
    Type:   | bytes[5] | bytes[12]   
    Dims:   | d1:3     |             
    Data:   |          |             
            | gene1    | cancer_genes
            | gene2    |             
            | gene3    | 


When loading data from databases or files, often slice names are assigned as given in the data source. In case of loading from Python data,
slice names are however autoassigned, and thus not very informative. To rename slices, one can use the :py:meth:`ibidas.representor.Representor.rename` function::

    >>> r.Rename("genes","scores")
    Slices: | genes    | scores
    
    >>> r.Rename(f1="scores")
    Slices: | f0   | scores

As this functionality is used often, a shorter version is available::

    >>> r/("genes","scores")
    Slices: genes    scores 
    
    >>> r/{f1:"scores"}
    Slices: f0    scores 
    
    >>> r.Get(_.f0/"genes", 
              Rep("cancer_genes")/"listname")
    Slices: genes       listname

Summary:
    * Slices can be compared to columns/fields in a table, or to data cursors which indicate on which data elements operations will be applied. 

    * A representor object is a collection of slices

    * Attribute lookup can be used to select a single slice.

    * More advanced selection can be performed using the ``get`` function, allowing multiple slice selection, slice modifications and slice combination. 


Dimensions
----------
Up to now, our data model was very similar to ones used in other software. *Dimensions* allow Ibidas to handle more complex data structures. 

Lets replace the first field of the table with a nested, variable length list::
    >>> data = [([1,2],0.5), ([3,4,5], 0.3), ([6,7,8,9], 0.8)]
    
    >>> r = Rep(data)
    Slices: | f0        | f1
    ----------------------------
    Type:   | int64     | real64
    Dims:   | d1:3<d2:~ | d1:3
    Data:   |           |
            | [1 2]     | 0.5
            | [3 4 5]   | 0.3
            | [6 7 8 9] | 0.8


Compare this to dims in the previous sections. Dimensions indicate which nesting arrays have been **unpacked**. So, in case of slice ``f0``, 
we are working at the level of ``int32`` (the type). These ``int32`` elements are nested two levels deep in arrays, which are indicated by 
dimension ``d1`` and ``d2``.  

The dim names are accompanied by a shape attribute. The ``3`` means that the dimension has a fixed size of 3, while the ``~`` means that the
dimension has a variable size. One might also encounter ``*``, which means that the dimension has a fixed, but unspecified, size. 

Dimensions are used in operations to determine how data is mapped w.r.t to each other::

    >>> r.f0 + r.f1
    Slices: | result               
    -------------------------------
    Type:   | real64               
    Dims:   | d1:3<d2:~            
    Data:   |                      
            | [ 1.5  2.5]          
            | [ 3.3  4.3  5.3]     
            | [ 6.8  7.8  8.8  9.8]

To perform an addition, both operands would need the same dimension normally. As this is not the case, 
we perform **broadcasting**. As you can see, elements from slice ``f1``  are broadcasted along dimension ``d2`` to enable 
the addition of the two slices. 


In case of type autodetection, dimension names are also assigned automatically (Starting from ``d1``). Dimension can however
be renamed in a similar way as slices, using :py:meth:`ibidas.representor.Representor.DimRename`::

    >>> r.DimRename("pathways","genes")
    Slices: | f0                 | f1        
    -----------------------------------------
    Type:   | int64              | real64    
    Dims:   | pathways:3<genes:~ | pathways:3
    Data:   |                    |           
            | [1 2]              | 0.5       
            | [3 4 5]            | 0.3       
            | [6 7 8 9]          | 0.8   

When performed without keywords, new dim names are mapped to dimensions by ordering dimensions on their nesting depth. If there are multiple
choices possible, an error will be returned. Similar to slice renaming, a shorthand is available using the ``%`` operator.


Summary:
    
    * The use of dimensions allow one to have slices with different dimensions within the same representor object

    * The use of broadcasting allows these slices to still interact (e.g. to perform a comparison or other operation)

    * Dimensions have a name and a shape

    * Dimensions are separated by the ``<`` symbol

    * Dimensions can be renamed using the ``DimRename`` function or the ``%`` shorthand operation. 


Dimension navigation
--------------------

Specifying how operations should be mapped to dimensions is done using two mechanisms.  The first
specifies at what type an operation should be executed (i.e. should an operation work on the  level of the list or the level of the list element?).
The second specifies across which dimension an operation should be executed (should we sum values along dimension d1 or d2?). 


The first mechanism is accessible through ``pack`` and ``unpack`` operations. 
There are two types of these operations:

    * pack/unpack from tuple type to slices and vice versa
    * pack/unpack from array type to dimensions and vice versa


Packing
~~~~~~~

The two most basic ``pack`` operations are respectively ``Tuple`` and ``Array``.

An example of the ``Array`` function::

    >>> data = [([1,2],0.5), ([3,4,5], 0.3), ([6,7,8,9], 0.8)]
    >>> r = Rep(data)

    >>> r.Array()
    Slices: | f0           | f1              
    -----------------------------------------
    Type:   | [d2:~]:int64 | [d1:3]:real64   
    Dims:   | d1:3         |                 
    Data:   |              |                 
            | [1, 2]       | [ 0.5  0.3  0.8]
            | [3, 4, 5]    |                 
            | [6, 7, 8, 9] |  


The influence of the array operation is that the dimensions are moved into the type. So subsequent operations
are performed at the level of the arrays::

    >>> r.Array().Get(_.f0 + _.f1)
    Slices: | result
    -------------------------------
    Type:   | [d2_d1:~]:real64
    Dims:   | d1:3
    Data:   |
            | [1 2 0.5 0.3 0.8]
            | [3 4 5 0.5 0.3 0.8]
            | [6 7 8 9 0.5 0.3 0.8

Note how an addition performed on arrays concatenates them. 

Arrays can also be packed with other aggregate operations. For example, the ``Set`` function::

    >>> r.f0.Set()
    Slices: | f0               
    ---------------------------
    Type:   | {sd2:~}<int64    
    Dims:   | d1:3             
    Data:   |                  
            | set([1, 2])      
            | set([3, 4, 5])   
            | set([8, 9, 6, 7])


    >>> r.f0.Set() | set([1])
    Slices: | f0
    ------------------------------
    Type:   | {sd2_d124:~}<int64
    Dims:   | d1:3
    Data:   |
            | set([1, 2])
            | set([1, 3, 4, 5])
            | set([8, 9, 1, 6, 7])

On sets, an or operation will take the union of two sets. 


Performing the ``Tuple`` operation gives::

    Slices: | data                               
    ---------------------------------------------
    Type:   | (f0=[d2:~]:int64, f1=real64)       
    Dims:   | d1:3                               
    Data:   |                                    
            | ([1, 2], 0.5)                      
            | ([3, 4, 5], 0.29999999999999999)   
            | ([6, 7, 8, 9], 0.80000000000000004)


As you can see, slice types are combined into a single slice tuple type. 

Summary:
    * Packing moves dimensions or slices into types
    * Dimensions can be packed using ``Array`` and ``Set``
    * Slices can be packed using ``Tuple``

Unpacking
~~~~~~~~~

The reverse operations for ``Array`` and ``Tuple`` are respectively ``Elems`` and ``Fields``::

    >>> r.Tuple().Fields()
    Slices: | f0        | f1    
    ----------------------------
    Type:   | int64     | real64
    Dims:   | d1:3<d2:~ | d1:3  
    Data:   |           |       
            | [1 2]     | 0.5   
            | [3 4 5]   | 0.3   
            | [6 7 8 9] | 0.8  

These are used less commonly as by default dat is unpacked by representation. 

Summary:
    * Unpacking unpacks types, moving the data structure into dimensions and slices
    * tuple/dictionary types can be unpacked using ``Fields``
    * array/set types can be unpacked using ``Elems``


Dimension selection
~~~~~~~~~~~~~~~~~~~

Operations such as ``Sum`` are 1-dimensional: that is, they operate across a single dimension, summing the elements. 
Normally, this is done on the last dimension::

    >>> r.Sum()
    Slices: | f0    | f1    
    ------------------------
    Type:   | int64 | real64
    Dims:   | d1:3  |       
    Data:   |       |       
            | 3     | 1.6   
            | 12    |       
            | 30    |     

Note how for both slices the last dimension has been collapsed (summed over). 

Of course, we cannot directly sum over dimension ``d1`` in slice ``f0``, as the elements do not line up. But in case of a matrix this is possible::

    >>> data = [([6,2],0.5), ([3,4], 0.3), ([6,4], 0.8)]
    >>> r = Rep(data)
    >>> r.Sum(dim='d1')
    Slices: | f0    | f1    
    ------------------------
    Type:   | int64 | real64
    Dims:   | d2:2  |       
    Data:   |       |       
            | 15    | 1.6   
            | 10    |       


Note how only dimension ``d2`` remains, dimension ``d1`` has been aggrated over by the ``Sum`` function. 
Other functions which can be used in this way include ``Max``, ``Min``, ``Argmax``, ``Argmin``, ``Any``, ``All``, ``CumSum``, ``Mean``, ``Sort``, ``Argsort``, ``Rank`` and ``Std``.


Broadcasting on dimension
-------------------------

We already saw some examples of broadcasting in action. A simple example is this one::

    >>> Rep([1,2,3]) + 3
    Slices: | data 
    ---------------
    Type:   | int64
    Dims:   | d1:3 
    Data:   |      
            | 4    
            | 5    
            | 6 

The value 3 is repeated along dimension ``d1`` to enable the (0-dimensional) addition operation. This 'repeating' is called 'broadcasting'. 

It also works with more complicated data structures. Assume that we want to normalize the arrays in ``r``, such that he mean value is equal to 0.0.

We can do that as follows::

    >>> m = r.Mean()
    Slices: | f0     | f1
    ---------------------------------
    Type:   | real64 | real64
    Dims:   | d1:3   |
    Data:   |        |
            | 4.0    | 0.533333333333
            | 3.5    |
            | 5.0    |

    >>> r - m
    Slices: | f0          | f1
    ----------------------------------------
    Type:   | real64      | real64
    Dims:   | d1:3<d2:2   | d1:3
    Data:   |             |
            | [ 2. -2.]   | -0.0333333333333
            | [-0.5  0.5] | -0.233333333333
            | [ 1. -1.]   | 0.266666666667


The Mean calculates the average value across the last dimension, i.e. dimension ``d2`` for slice ``f0``, and dimension ``d1`` for slice ``f1``. Using broadcasting,
we can directly subtract this from the whole dataset. 


If we now want to normalize only across dimension ``d1``, this can be simply accomplished using::

    >>>  r - r.Mean(dim='d1')

    Slices: | f0                        | f1
    ------------------------------------------------------
    Type:   | real64                    | real64
    Dims:   | d1:3<d2:2                 | d1:3
    Data:   |                           |
            | [ 1.         -1.33333333] | -0.0333333333333
            | [-2.          0.66666667] | -0.233333333333
            | [ 1.          0.66666667] | 0.266666666667


To also divide by the standard deviation, we simply add::

    >>> (r - r.Mean(dim='d1')) / r.Std(dim='d1')

A simple shortcut is::
    
    >>> Alg.scaling.Whiten(r,dim='d1')


Broadcasting works by matching dimensions in the operands to each other, expanding dimensions that are not available in either operand. 

There are some noteworthy special cases. 

First, the ordering of the dimensions. Suppose we have a dataset::

    >>> data = [([6,2],[0.5]), ([3,4], [0.3,0.4]), ([6,4], [0.8,0.2])]
    >>> r = Rep(data)
    Slices: | f0        | f1
    ---------------------------------
    Type:   | int64     | real64
    Dims:   | d1:3<d2:2 | d1:3<d3:~
    Data:   |           |
            | [6 2]     | [ 0.5]
            | [3 4]     | [ 0.3  0.4]
            | [6 4]     | [ 0.8  0.2]

What happens if we add f0 and f1? ri


The first is the case in which dimensions in both operands are available, but not ordered correctly. E.g. can we add something with dimensions ``a<b`` to a slice with dimensions ``b<a``? 
Ibidas does not reorder dimensions in such cases. So, suppose we have::

    >>> r.f0
    Slices: | f0       
    -------------------
    Type:   | int64    
    Dims:   | d1:3<d2:2
    Data:   |          
            | [6 2]    
            | [3 4]    
            | [6 4]  


    >>> r.f0.Transpose()
    Slices: | f0       
    -------------------
    Type:   | int64    
    Dims:   | d2:2<d1:3
    Data:   |          
            | [6 3 6]  
            | [2 4 4] 


    >>> r.f0 + r.f0.Transpose()
    Slices: | f0                     
    ---------------------------------
    Type:   | int64                  
    Dims:   | d2:2<d1:3<d2:2         
    Data:   |                        
            | [[12  8] [6 7] [12 10]]
            | [[8 4] [7 8] [10  8]]  

(Note that the Transpose operation reverses the dimension ordering)

As you can see, the output is ``b<a<b``. The rules for these types of broadcasting are as follows::
    * one starts with the rightmost operand, and the last dimension
    * one maps this dimension where possible to the other operands, taking the first matching dimension up from the most nested dimension. 
    * next, one maps the next dimension in the rightmost operand. However, one only looks further upward from the last matched dimenson in the other operand. 
    * if a dimension cannot be matched, it is broadcasted.

E.g. in this case one start with dimension ``d1`` in the right operand, this one is matched to the dimension ``d1`` in the left operand. The next dimension in the
rightmost operand (``d2``) cannot be matched in the leftmost operand, as there is no more upward dimension than ``d1``, so it is broadcasted. The rightmost operand
is finished, so we move on to the next operand, and as the first d2 dimension is not yet matched, we broadcast it to the rightmost operand. 

Due to the ordering dependence of operands, this means that one can influence the dimension ordering by rearranging operands. For example::
    
    >>> r.f0.Transpose() + r.f0
    Slices: | f0                  
    ------------------------------
    Type:   | int64               
    Dims:   | d1:3<d2:2<d1:3      
    Data:   |                     
            | [[12  9 12] [4 6 6]]
            | [[9 6 9] [6 8 8]]   
            | [[12  9 12] [6 8 8]]

Now the ordering is ``a<b<a``. In reality, these situations in which the operand position matters do not occur that often.


Summary:
    * Broadcasting maps dimensions in operands to each other, repeating across dimensions that do not occur in any of the operands

    * Broadcasting does not reorder dimensions. 

    * In some cases, the ordering of the operands can influence the dimension ordering. 

    

Broadcasting on position
------------------------

Ibidas matches operands normally on dimension identity. In cases these dimensions do not match, they are broadcasted.
This means that if one has:




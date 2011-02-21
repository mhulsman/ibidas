Basic concepts
==============

.. warning::
   This section is a bit outdated, please use the tutorial.

In this section we describe some of the basic underlying concepts of Ibidas.

Representation
--------------

In Ibidas one works with data sources by packaging them within a 'representation object'.

Such data sources can be python objects, files, databases, and so on. 
Once they are wihtin a representation object they are all handled similarly.

We start with a simple example, in which we package an integer object::

    >>> rep(3)
    Slices: data  
    Types:  int64
    Dims:         
    
    Data: 3

.. tip::
    The concepts ``slices``, ``types`` and ``dims`` will be explained in the next sections


One can perform all kind of operations with data that is in a representation object, e.g::

    >>> rep([1,2,3]) + 3
    Slices: data  
    Types:  int64 
    Dims:   d1:3  

    Data: [4 5 6]

Summary:
    * A representor object encapsulates a data source. 

    * Data sources can be python objects, but also files or databases.

Lazy execution
--------------

You might have noted that executing the previous commands resulted immediatly in a printout 
of the contents of the representation object. This is due to the IPython interpreter, 
which will print a representation of the result of all non-assignment operations. 
So, if we instead would have executed::
    
    >>> r = rep([1,2,3]) + 3

no output would have been printed. More importantly however, Ibidas would also not have performed the
requested addition operation. I.e. ``r`` would have only been a representation of the requested operations 
and data sources, and thus not the result of those operations.

.. important::
    Operations are executed lazily. I.e. only when output is requested by the user.

The reason for this behaviour is that it allows optimizations which are otherwise not possible. For example, it 
enables the system to translate queries against a database (partially) into the query language SQL. This way, 
instead of moving unnecessary data to Ibidas, operations can be moved to the database.

.. note::
   For brevity, we will sometimes print output in this tutorial after assignments (e.g. ``r = rep(3)``.
   In reality this does not happen. One can still get the output after such statements, by simply execting ``r``

Often, one wants to just see a description of the contents of a representation object, not the actual data result itself.
This can be done using the information attribute ``I``::

    >>> r = rep(["geneA","geneB"])
    >>> r.I
    Slices: data     
    Types:  bytes[5] 
    Dims:   d1:2 

Note that the data is not printed. Especially in case of slow operations or data sources this can be useful.


Getting data out of a representor object is simple, one simply appends ``()`` to a query to let it return the 
results as normal ``numpy`` or python objects::

    >>> r()
    array(['geneA', 'geneB'], dtype='|S5')

As you can see, Ibidas has packaged the data in a numpy array. 

Summary:
    * Operations are only executed when needed, to allow for optimizations

    * One can ask for a description of the representor contents using the ``I`` attribute.

    * One can get the data results by transforming the query into a function call by appending ``()``

Types
-----
When one executes ``rep`` without specifying a type, the type is detected automatically.
For example, in the first example, the detect type was ``int64``. 

.. note::
    Depending on the platform you use, the type can also be ``int32``.

The type determines how operations on the representor are handled.
For example, with an integer type, one can perform standard integer operations on the representor::
    
    >>> r = rep(3)
    >>> (r + 3) * r
    Slices: data 
    Types:  int32 
    Dims:         

    Data: 18

Similarly, in case of the string type, the addition operation becomes concatenation::

    >>> rep(["geneA", "geneB"])  + "_geneC"
    Slices: data      
    Types:  bytes[11] 
    Dims:   d1:2      

    Data: ['geneA_geneC' 'geneB_geneC']

One might have noted that, although we now represent a list of thins, the type still represents the
type of the list elements. 

This is because ``rep`` (by default) **unpacks** the data. By unpacking, operations
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
    
    >>> r = rep(data)
    Slices: f0       f1     
    Types:  bytes[5] real64 
    Dims:   d1:3     .      

    Data: (array(['gene1', 'gene2', 'gene3'], 
        dtype='|S5'), array([ 0.5,  0.3,  0.8]))


If we compare this to earlier output, we now see that there are more than one columns in the data description.

These columns represented *slices*. Slices are one of the main concepts in Ibidas. They can be compared to columns/fields in a table, but are more general.

Selecting a slice can be done using simple attribute lookup::

    >>> r.f0
    Slices: f0       
    Types:  bytes[5] 
    Dims:   d1:3     

    Data: ['gene1' 'gene2' 'gene3']


Each slice has a name (the first row), a type (second row) and a dimension (third row). Dimensions will be explained later. For now, it is important that each
slice has a common type. This means that all data elements adressed by it can be handled in the same way. Slices could thus also be seen as a kind of cursor in your data structure.
Performing operations on this cursor will perform the operations on a subset of your data. For example::

    >>> r.f0 == "gene2"
    Slices: f0
    Types:  bool
    Dims:   d1:3

    Data: [False  True False]

To select multiple slices, one can use the :py:meth:`ibidas.representor.Representor.get` function::

    >>> r.get("f1", "f0")
    Slices: f1     f0       
    Types:  real64 bytes[5] 
    Dims:   d1:3   . 

    >>> r.get(1, 0)
    Slices: f1     f0       
    Types:  real64 bytes[5] 
    Dims:   d1:3   . 

    >>> r.get(r.f1, r.f0)
    Slices: f1     f0       
    Types:  real64 bytes[5] 
    Dims:   d1:3   . 

As you can see, there are multiple options to address slices.  The third option is useful, as this can also be combined with other operations::

    >>> r.get(r.f1 + 3, r.f0)
    Slices: f1     f0       
    Types:  real64 bytes[5] 
    Dims:   d1:3   .        

    Data: (array([ 3.5,  3.3,  3.8]), array(['gene1', 'gene2', 'gene3'], 
        dtype='|S5'))

One can also use this function to combine slices, e.g::

    >>> r.get(r.f0, rep("cancer_genes"))
    Slices: f0       data
    Types:  bytes[5] bytes[12]
    Dims:   d1:3

    Data: (array(['gene1', 'gene2', 'gene3'],
        dtype='|S5'), 'cancer_genes')


When loading data from databases or files, often slice names are assigned as given in the data source. In case of loading from Python data,
slice names are however autoassigned, and thus not very informative. To rename slices, one can use the :py:meth:`ibidas.representor.Representor.rename` function::

    >>> r.rename("genes","scores")
    Slices: genes    scores 
    
    >>> r.rename(f1="scores")
    Slices: f0    scores 

As this functionality is used often, a shorter version is available::

    >>> r/("genes","scores")
    Slices: genes    scores 
    
    >>> r/{f1:"scores"}
    Slices: f0    scores 
    
    >>> r.get(r.f0/"genes", 
              rep("cancer_genes")/"listname")
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
    
    >>> r = rep(data)
    Slices: f0        f1     
    Types:  int32     real64 
    Dims:   d1:3<d2:~ .      

    Data: (array([[1 2], [3 4 5], [6 7 8 9]], dtype=object), array([ 0.5,  0.3,  0.8]))

Compare this to dims in the previous sections. Dimensions indicate which nesting arrays have been **unpacked**. So, in case of slice ``f0``, 
we are working at the level of ``int32`` (the type). These ``int32`` elements are nested two levels deep in arrays, which are indicated by 
dimension ``d1`` and ``d2``.  

The dim names are accompanied by a shape attribute. The ``3`` means that the dimension has a fixed size of 3, while the ``~`` means that the
dimension has a variable size. One might also encounter ``*``, which means that the dimension has a fixed, but unspecified, size. 

So what does the ``.`` mean in the dimension of slice ``f1``? In case of large tables with many slices and long dimension names, it can be a bit
unwieldy to repeat the same dimensions for every slice. Therefore, we use a shorthand. The ``.`` here means that it has the same first dimension as the 
previous slice.  If it would be ``.<.``, it would mean that it shares the first two dimensions, and so on. 

Dimensions are used in operations to determine how data is mapped w.r.t to each other::

    >>> r.f0 + r.f1
    Slices: result    
    Types:  real64    
    Dims:   d1:3<d2:~ 

    Data: [[ 1.5  2.5] [ 3.3  4.3  5.3] [ 6.8  7.8  8.8  9.8]]

To perform an addition, both operands would need the same dimension normally. As this is not the case, 
we perform **broadcasting**. As you can see, elements from slice ``f1``  are broadcasted along dimension ``d2`` to enable 
the addition of the two slices. 


In case of type autodetection, dimension names are also assigned automatically (Starting from ``d1``). Dimension can however
be renamed in a similar way as slices, using :py:meth:`ibidas.representor.Representor.dim_rename`::

    >>> r.dim_rename("pathways","genes")
    Slices: f0                 f1
    Types:  int32              real64
    Dims:   pathways:3<genes:~ .
    
    >>> r.dim_renmae(d1="pathways")
    Dims:   pathways:3<d2:~ .      

When performed without keywords, new dim names are mapped to dimensions by ordering dimensions on their nesting depth. If there are multiple
choices possible, an error will be returned. Similar to slice renaming, a shorthand is available using the ``%`` symbol.


Summary:
    
    * The use of dimensions allow one to have slices with different dimensions within the same representor object

    * The use of broadcasting allows these slices to still interact (e.g. to perform a comparison or other operation)

    * Dimensions have a name and a shape

    * The printout of a representor uses ``.`` to indicate a common dimension w.r.t to the previous slice.

    * Dimensions are separated by the ``<`` symbol

    * Dimensions can be renamed using the dim_rename function or the ``%`` shorthand operation. 


Packing/unpacking
-----------------
Navigating accross dimensions and slices can be performed with ``pack`` and ``unpack`` operations. 
There are two types of these operations:

    * pack/unpack from tuple type to slices and vice versa
    * pack/unpack from array type to dimensions and vice versa


Packing
~~~~~~~

The two most basic ``pack`` operations are respectively ``tuple`` and ``array``.

An example of the ``array`` function::

    >>> data = [([1,2],0.5), ([3,4,5], 0.3), ([6,7,8,9], 0.8)]
    >>> r = rep(data)

    >>> r.array()
    Slices: f0           f1            
    Types:  [d2:~]:int32 [d1:3]:real64 
    Dims:   d1:3                       

    Data: (array([[1 2], [3 4 5], [6 7 8 9]], dtype=object), array([ 0.5,  0.3,  0.8]))


The influence of the array operation is that the dimensions are moved into the type. So subsequent operations
are performed at the level of the arrays. 

Arrays can also be packed with other aggregate operations. For example, the ``set`` function:

    >>> r.f0.set()


Performing the ``tuple`` operation gives::

    >>> r.tuple()
    Slices: data                         
    Types:  (f0=[d2:~]:int32, f1=real64) 
    Dims:   d1:3                         

    Data: [(array([1, 2]), 0.5) (array([3, 4, 5]), 0.29999999999999999)
    (array([6, 7, 8, 9]), 0.80000000000000004)]

As you can see, slice types are combined into a single slice tuple type. 

Unpacking
~~~~~~~~~

The reverse operations for ``array`` and ``tuple`` are respectively ``elements`` and ``attributes``.

These are used less commonly as by default dat is unpacked by representation. 



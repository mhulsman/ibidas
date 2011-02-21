Tutorial
========

In/out-degree distribution of a transcriptional network
--------------------------------------------------------------------------

In the first example, we will look at the network properties of the yeast transcriptional network.
For that purpose, we obtain regulations from www.yeastract.com > Flat Files > RegulationTwoColumnTable

Loading the data
~~~~~~~~~~~~~~~~
To get the data into Ibidas we perform the following operations::

    >>> url = "http://www.yeastract.com/download/RegulationTwoColumnTable_Documented_20101213.tsv.gz"
    >>> yeastract = Read(Fetch(url))

``Fetch`` is an operation that takes a url, downloads the file to the data directory (default ~/.ibidas/data/),
assigns it a file name, and returns this filename. Executing ``Fetch`` on this url again will just return the filename of the
cached local copy. 

Subsequently, ``Read`` takes this filename, and imports the data. By
default is assumes a comma/tab/other symbol separated value file. It will attempt to detect the column delimiter, 
availability of fieldnames and existence of comment lines automatically. 

After these two operations, the contents of yeastract look like this::    
    >>> yeastract
    Slices: | f0       | f1      
    -----------------------------
    Type:   | bytes[7] | bytes[9]
    Dims:   | d1:48082 | d1:48082
    Data:   |          |         
            | Abf1     | YKL112w 
            | Abf1     | YAL054c 
            | Abf1     | YGL234w 
            | ...      | ...


Note that this file did not have fieldnames, so slice names as well as slice types and slice dimensions (will be explained later) where 
determined automatically. The left column (slice ``f0``) contains transcription factors, while the right column (slice ``f1``) 
contains the targets. Note that transcription factor `Abf` is repeated multiple times in this file, as it has multiple targets. 


Looking at the types of both slices, we see that this has been detected as the ``bytes`` type. 
Furthermore, the ``bytes`` type has an extra specifier ``[7]``, indicating that this is the max length of the entries in the corresponding slice.

.. tip:: the ``bytes`` type corresponds to the default ``str`` type in Python 2.x. In Python 3.0 the default string type has become
   unicode, and ``str`` has been renamed to ``bytes``. In Ibidas we use the same naming scheme: ``bytes`` for the old str type, and
   ``string`` for unicode strings. 



The third attribute, Dims, describes the structure of the data in a slice. We see that both slices have a single dimension.
This means that the data is structured in a way similar to a `vector`. A slice without dimensions would correspond to a `scalar` value, 
while a slice with two fixed dimensions would correspond to a `matrix` structure. In this case, both slices share the same dimension ``d1``, indicating that
the shape and ordering of elements in both slices are shared. The ``:48082`` describes the shape of the dimension, indicating
that there are 48082 rows.

Of course, using meaningless names, such as ``f0``, is not very handy, so we first assign more suitable names::
    
    >>> yeastract = yeastract/('trans_factor','target')

While we are at it, we also assing a more suitable dimension name::

    >>> yeastract = yeastract%'tftargets'

Note that we reuse the division operator and the modulo operator to assing these names. The reason for this is that 
naming slices is an relatively often recurring action. It is however also possible to use instead a regular function, e.g::

    >>> yeastract = yeastract.Rename('trans_factor','target')

Renaming slices and dimensions is not always necessary, many data sources already assign names of the various data elements. E.g. databases have a table name which
can be used as dimension name, while the column names can be used as a slice names.

.. tip:: You may have noted that operations in Ibidas start with an uppercase letter. The reaseon for this is to separate them from
   slice names and dimension names, which have to be always in lower case. I.e. data.count would refer to the slice count, while
   data.Count refers to the operation ``Count``.


After renaming, our data now looks like::
    >>> yeastract
    Slices: | trans_factor    | target
    -------------------------------------------
    Type:   | bytes[7]        | bytes[9]
    Dims:   | tftargets:48082 | tftargets:48082
    Data:   |                 |
            | Abf1            | YKL112w
            | Abf1            | YAL054c
            | Abf1            | YGL234w
            | ...             | ...

You might have found that there is a very small delay before the contents are printed. The reason for this is the lazy nature of 
Ibidas. That is, Read and all subsequent operations are not executed immediatly, but only when asked for results. The reason
for this is that it allows for improved optimizations. For example, in case the data sources is a database, it allows us to translate operations
into SQL. 

In this case however, we would like to perform all operations and store their result. For that purpose, their is the operation ``Copy``::

    >>> yeastract = yeastract.Copy()

Now our data source is ready for use. 

Predefined data sources
~~~~~~~~~~~~~~~~~~~~~~~

Before actually using the transcription factor data, we make a sidestep. Performing the actions we just did each time you want to load
the yeastract data source is a bit cumbersome. For this reason, Ibidas allows the defenition of predefined data source loading functions.

For yeastract we have added just such a function which performs the actions we have just gone through. So instead of performing those
operations, we could also just execute::
    
    >>> yeastract = Get.yeastract()

The data source functions are found in ibidas/pre.py. One can easily add new data sources. For example, 
adding the yeastract data resource yourself, would have required the following code::

    from ibidas import *

    def yeastract(url="http://www.yeastract.com/download/RegulationTwoColumnTable_Documented_20101213.tsv.gz"):
        """Downloads documented transcription factor regulation interactions from yeastract"""

        res = Read(Fetch(url),dtype="[tftargets:*]<(trans_factor=bytes, target=bytes)")
        return res.Copy()

    Get.register(yeastract)

This can be simply put into a file which can then be imported when needed. These functions can also be shared with others.
In fact, if it is a public data resource, one is encouraged to submit it for inclusion into Ibidas itself. 

.. note::
   In the implementation as shown, we took a slightly different approach to specifiying slice names and dimension names then used
   in the previous section. Here, we specify it through the data type. Specifically, it reads as having an array ``[tftargets:*]`` 
   of unspecified size,  with as elements ``(trans_factor=bytes, target=bytes)`` tuples of ``trans_factor`` and ``target`` fields, 
   both with type ``bytes``. This array of tuples is automatically unpacked into the structure we encountered in the previous section.
   By specifying the whole type at forehand, no automatic type detection is needed when loading the file. 

Calculating in/out degree
~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have access to the data sources, we will determine the in/out-degree. This is done by calculating
respectively the number of transcription factors per common target and the number of targets per transcription factor.

For this, we will use the GroupBy operation. First, to show what the GroupBy operation does, we will Group the data on the 
transcription factors::

    >>> gyeastract = yeastract.GroupBy(_.trans_factor)

.. note::
    Note that we could also have written the command as::

        >>> gyeastract = yeastract.GroupBy(yeastract.trans_factor)

    Attribute access is used to obtain single slices. However, this quickly becomes cumbersome in large expressions with multiple operations. Therefore,
    we use what we call a `context` operator _.  This operator functions as replacement for the enclosing data object. We will show more uses of this operator
    further on. 

Now, lets look at what this operation does::
    >>> gyeastract
    Slices: | trans_factor    | target                                                                        
    ----------------------------------------------------------------------------------------------------------
    Type:   | bytes           | bytes                                                                         
    Dims:   | gtrans_factor:* | gtrans_factor:*<gtftargets:~                                                  
    Data:   |                 |                                                                               
            | Hir2            | [YDR101c YKL109w YBR009c YNL030w YBR010w YNL031c YKL110c YGR233c YDR103w;  YG~
            | Zap1            | [YMR120c YGL256w YOR134w YNL336w YBR302c YML132w YFL062w YGR295c YHL048w;  YD~
            | Pip2            | [YPR128c YML042w YNR001c YOR100c YOR180c YLR284c YER015w YKR009c YLR174w;  YO~
            | ...             | ... 

After grouping on transcription factors, there is for each transcription factor now a single row. All
targets corresponding to the transcription factor have been gathered in a nested array. This is reflected 
in the metadata. Although the types of the slices are still the same, the dimensions have changed. 

Slice ``trans_factor`` now has a new dimension, called `gtrans_factor:*`. This name is made from concatenating `g` (from group) with the
name of the slice on which we performed the grouping. We see that the shape ``*`` is undefined. This is because
at forehand, it was not known how many rows this operation would return. 

Slice ``target`` now similarly has the `gtrans_factor:*` dimension, but it has also gained an extra dimension, called `gtftargets:~`. This is due to the grouping,
which did put targets into nested arrays, corresponding to each transcription factor. Note that the shape paremeter here is `~`, 
unlike the `*` in the `gtrans_factor` dimension. This indicates that the shape of this dimension is not fixed, but variable. 
So, although previously we said that having two fixed dimensions would lead to a matrix, one can thus also have variable dimensions. This allows us to handle
nested data and multi-dimensional data in a similar way. 


Now, to obtain the out-degree of the transcription factors, we simply have to count the size of the nested arrays::
    >>> gyeastract = gyeastract.Get(_.trans_factor, _.target.Count()/"out_degree")
    Slices: | trans_factor    | out_degree
    -------------------------------------------
    Type:   | bytes           | int64
    Dims:   | gtrans_factor:* | gtrans_factor:*
    Data:   |                 |
            | Hir2            | 68
            | Zap1            | 185
            | Pip2            | 150
            | ...             | ...

Here, we introduce two new operations, ``Get`` and ``Count``. Starting with the last one, ``Count`` is an aggregation operator which takes
the last dimension, and counts the number of elements in it. We subsequently call the resulting slice ``out_degree``. 
Note that its second dimension has been collapsed into the count. 

The ``Get`` function is similar to the SELECT phrase in SQL. It allows one to select some slices from a dataset, perform on some of them operations, 
and then returns the combined object. As is shown, it accepts (among other things) context operators to specifiy which slices should be selected and 
which actions performed. 


Now, if we want to plot this distribution we can make use of ``matplotlib``. For that, we have to get the data out of the data object. This
can be very simply done by making the query a call by adding ``()`` to it::
    
    >>> from matplotlib.pylab import *
    >>> hist(gyeastract.out_degree())
    >>> show()

Now, to plot the in-degree distribution we can do something similar. The total script becomes::

    >>> from matplotlib.pylab import *
    >>> yeastract = Get.yeastract()

    >>> subplot(211)
    >>> hist(yeastract.GroupBy(_.trans_factor).target.Count()(), bins=50)
    >>> title('Out degree')

    >>> subplot(212)
    >>> hist(yeastract.GroupBy(_.target).trans_factor.Count()(), bins=50)
    >>> title('In degree')

    >>> show()

Resulting in the following image:

.. image:: inout_dist.png


Chromosome locations
--------------------

Next, suppose we want to analyze the genomic locations of the targets. For that purpose, we need for all genes the location on the chromosomes.


Loading the data
~~~~~~~~~~~~~~~~
This can be found in the ``SGD_features.tab``, which can be obtained from yeastgenome.org. We use the same strategy to load this file. Unfortunately, 
also this file comes without fieldnames, so we specify those through the type::

    rtype = """[feats:*]<(sgdid=bytes, feat_type=bytes, feat_qual=bytes, feat_name=bytes, gene_name=bytes, 
                          gene_aliases=bytes, feat_parent_name=bytes, sgdid_alias=bytes, chromosome=bytes, 
                          start=bytes, stop=bytes, strand=bytes[1], genetic_pos=bytes, coordinate_version=bytes[10], 
                          sequence_version=bytes, description=bytes)"""

    res = Read(Fetch("http://downloads.yeastgenome.org/chromosomal_feature/SGD_features.tab"),dtype=rtype)

Note that one could also just have named the fields that were needed, for example using::

    res = res/{'f3': 'feat_name', 'f8':chromosome, 'f9':start}

When reading a file like this one, all input data is in string(bytes) format. First, we cast the necessary fields
to other types::

    res = res.To(_.start, _.stop, Do=_.Cast("int$"))
    res = res.To(_.genetic_pos,   Do=_.Cast("real64$"))

Here we introduce two new operations. To is a utility function, which allow one to apply other operations to a subselection 
of the slices in a data set. In this case, we cast the ``start`` and ``stop`` slice each to integer, and the ``genetic_pos``
slice to a double floating point type. This is what the ``Cast`` operation does. Note that we do specify ``int$``, i.e. with a
dollar sign. The dollar sign here means that missing values (empty fields) are allowed. 

Next, we take a look at the ``gene_aliases`` field, which contains multiple gene aliases separated by the '|' symbol.
To split this into a nested array, we use the split function::

    res = res.To(_.gene_aliases,  Do=_.Each(_.split('|')).Elem()[_ != ""])

Here, we introduce three new functions. The ``Each`` function applies a regular python function or a context object to each
element in a slice. In this case, we split each string into a list of strings using the _.split('|') operation. The slice returning
from this has arrays as the operative type. As we want to operate on the individual gene names, we use the Elem() function, which
`unpacks` this array, such that subsequent operations will be performed on the elements instead of the arrays of elements. 
Lastly, we apply a filter operation, removing all empty gene names from the gene names lists. 

Note that Ibidas does not know what type will result from the function used in the ``Each`` operation. For that reason it will automatically
perform type detection when necessary for subsequent operations. It is possible to prevent this by specifying the type at forehand. Also, instead
of the context operation one can use functions, which are slightly faster than context oeprations::
    
    splitfunc = lambda x: x.split('|')
    res = res.To(_.gene_aliases,  Do=_.Each(splitfunc, dtype="[aliases:~]<bytes").Elem()[_ != ""])

As last step, we execute all operations, and store the result in memory::
    yeast_feats = res.Copy()


Note that this dataset is also predefined in Ibidas, and can be obtained using::

    yeast_feats = Get.yeast_feats()


Linking the datasets
~~~~~~~~~~~~~~~~~~~~

Now, we have to match both datasets. This is done on the ``targets`` in the Yeastract dataset, and on the ``feat_name`` field in the SGD_features dataset.
However, both fields use different strategies for uppercase/lowercase, so first we have to change both to use always upper case.
The total operation now becomes::
    
    >>> tftargetdat = yeastract.Match(yeast_feats, _.target.Each(str.upper), _.feat_name.Each(str.upper))
    
    >>> tftargetdat  #only showing the first few slices...
    Slices: | trans_factor     | target           | sgdid            | feat_type        | feat_qual       
    ------------------------------------------------------------------------------------------------------
    Type:   | bytes            | bytes            | bytes            | bytes            | bytes           
    Dims:   | tftargets_feats~ | tftargets_feats~ | tftargets_feats~ | tftargets_feats~ | tftargets_feats~
    Data:   |                  |                  |                  |                  |                 
            | Gcr2             | YAL008w          | S000000006       | ORF              | Verified        
            | Met4             | YAL008w          | S000000006       | ORF              | Verified        
            | Otu1             | YAL008w          | S000000006       | ORF              | Verified        
            | ...              | ...              | ...              | ...              | ...             


This operation will links rows in yeastract with rows in yeast_feats, based on equality in the ``target`` and ``feat_name`` column. Any ``target`` row for which
no entry can be found in ``feat_name`` will be left out. We do a quick check to determine how many of the rows could not be matched::
    
    >>> yeastract.target.Count() - tftargetdat.target.Count()
    Slices: | target
    ----------------
    Type:   | int64 
    Dims:   |       
    Data:   |       
            | 72

This means that 72 transcription factor-target pairs could not be matched. On 48010 pairs this is negligible. However, as this is a tutorial, we will look into this
a bit more thoroughly. First, we determine which targets where not matched::


    >>> non_matched = (yeastract.target.Each(str.upper).Set() - tftargetdat.target.Each(str.upper).Set()).Elem()
    >>> non_matched
    Slices: | target
    ---------------------------------------
    Type:   | bytes[9]
    Dims:   | stftargets_stftargets_feats:*
    Data:   |
            | YLR157W-C
            | YAR044W
            | YBL101W-C
            | YBL101W-A
            | YJL017W
            | A1
            | YJL012C-A
            | MALT
            | MALS
            | SNR20
            | A2
            | RDN5
            | ALD1
            | YDR474C
            | YBR075W
            | TER1
            | SUC6
            | YDR524W-A
            | YGR272C
            | YDL038C
            | YBL101W-B
            | DEX2



This introduces the ``Set`` command. Using the set command, the elements of the (by default last) dimension are packed into a set. A set is a collection of objects
in which each element is unique. That is, adding the string "YLR157W-C"  multiple times to a set will result in a set with just one occurence of "YLR157W-C".
Sets have some special operations defined on them. One of them is set substraction, which was used here. It removes all elements in the set of the first operand that
also occur in the set of the second operand, leaving only the elements that do not occur in the second operand. In this case thus the elements that were not matched. 

Next, we use the ``Elem`` operation to unpack the resulting set, and ``Show`` to see the whole result. 

The names in the list suggest that we might find matching rows by looking either at the ``gene_name`` or ``gene_aliases`` column. The ``gene_name`` column gives no match however::
    
    >>> non_matched.In(feats.gene_name.Each(str.upper))



Next we look at the gene_aliases column. 






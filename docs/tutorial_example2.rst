Chromosome distribution
=======================

In this example, we will analyze the genomic locations of transcription factor targets. We will determine if transription 
factors favor specific chromosomes. Also, it would be interesting to determine if certain chromosomes are common in terms of transcription factors,
and/or transcription factors common in the chromosomes they target. 


Importing the data
~~~~~~~~~~~~~~~~~~
We will use the transcription factor data that has been imported in the previous example. The same data 
can be obtained directly using::

    >>> yeastract = Get.yeast.yeastract()

Next to the transcription factor data, we need the location of all genes on the chromosomes.
This information can be found in the ``SGD_features.tab``, which can be obtained from yeastgenome.org. 

Unfortunately, similar to yeastract, also this file comes without fieldnames, so we specify those through the type::

    rtype = """[feats:*]<(sgdid=bytes, feat_type=bytes, feat_qual=bytes, feat_name=bytes, gene_name=bytes, 
                          gene_aliases=bytes, feat_parent_name=bytes, sgdid_alias=bytes, chromosome=bytes, 
                          start=bytes, stop=bytes, strand=bytes[1], genetic_pos=bytes, coordinate_version=bytes[10], 
                          sequence_version=bytes, description=bytes)"""

    res = Read(Fetch("http://downloads.yeastgenome.org/chromosomal_feature/SGD_features.tab"),dtype=rtype)

Note that, instead of specifying the type, we could also just have named the slicees that were needed, for example using::

    res = res/{'f3': 'feat_name', 'f8':'chromosome', 'f9':'start'}

This would rename field 3, 8 and 9 (starting from 0!). 

Type casting
^^^^^^^^^^^^
While not all fields will be used in this example, for the purpose of the tutorial we will attempt to prepare the whole dataset for easy use. 

First, when reading a file like this one, all input data is in string(bytes) format. For some slices this is not the ideal format.
Therefore, we change the types of certain slices from ``bytes`` to ``int`` and ``real`` types. This is an operation that is known as casting::

    res = res.To(_.start, _.stop, Do=_.Cast("int?"))
    res = res.To(_.genetic_pos,   Do=_.Cast("real64?"))

``To`` is a utility function, which allow one to apply other operations to a subselection 
of the slices in a data set. In this case, we cast the ``start`` and ``stop`` slice to an integer type, and the ``genetic_pos``
slice to a double floating point type. Note that we do specify ``int?``, i.e. with a
question mark sign. The indicates that missing values (empty fields) are allowed. 

.. note:: 
    Maybe you ask yourself why we do not use the following approach::
        
        >>> res.genetic_pos = res.genetic_pos.Cast("real64?")

    The reason for that is that res could have been used in another query before executing this command. Changing res by 
    performing this operation would therefore lead to some problems because of the lazy nature of query execution in Ibidas.
    It might be possible to allow this in the future, however it would require some trickery. So, for now, we use the approach
    with the ``To`` operation.
    

Applying a regular Python function and filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, we take a look at the ``gene_aliases`` slice, which contains multiple gene aliases separated by the '|' symbol.
We would like to split these strings into individual names, and remove the empty names. For the split operation, we 
use the standard Python split function. The whole expression becomes::

    >>> splitfunc = _.split('|')
    >>> res.gene_aliases.Each(splitfunc).Elems()[_ != ""]

`splitfunc` is here a context operator based expression, which can be applied to a string in order to split it. 

``Each`` applies a regular python function or a context object to each element in a slice. The slice returning from this has 
in this case as type lists of strings, as that is the output of the splitfunc operation. 

``Elems`` `unpacks` this resulting list of names, such that subsequent operations will be performed on the list elements instead of the list itself. 

``Filter``, denoted by the `[]`, only keeps elements (denoted by the context operator) that are unequal to the empty string. 

.. note::
    Note that Ibidas cannot know what type will result from the function used in the ``Each`` operation. For that reason it will automatically
    perform type detection when necessary for subsequent operations. It is possible to prevent this by specifying the type at forehand. 
    Also, instead of the context operation one can use regular python functions, which (at the moment) execute slightly faster::
        
        >>> splitfunc = lambda x: x.split('|')
        >>> dtype = "[aliases:~]<bytes"
        >>> res.gene_aliases.Each(splitfunc, dtype=dtype).Elems()[_ != ""]

    (lambda allows one to define anonymous functions in Python)

To make these modified gene_aliases slice part of the dataset, we apply them again using the ``To`` function, and store the results using ``Copy``::

    splitfilter = _.Each(splitfunc, dtype=dtype).Elems()[_ != ""]
    yeast_feats = res.To(_.gene_aliases, Do=splitfilter).Copy()


Short version
^^^^^^^^^^^^^

To obtain both datasets directly, use::

    yeast_feats = Get.yeast.genomic_feats()
    yeastract = Get.yeast.yeastract()


Linking the datasets
~~~~~~~~~~~~~~~~~~~~

Now, we have to link both the yeastract dataset and the genomic features dataset. This is done by matching the ``targets`` in the Yeastract dataset 
with the ``feat_name`` slice in the genomic features dataset. This can be accomplished using the ``Match`` operation, which links rows in two datasets
based on equality of the entries in two slices.

For example, we could use::

    >>> tf_feat = yeastract |Match(_.target, _.feat_name)| yeast_feats

to match both datasets on their target and feat_name slice. 

However, there is the small problem that both datasets have different upper/lowercase usage, due to which
most target and feat_name names do not match with each other. 

So, instead, we convert each target and feat_name to upper case before matching::
    
    >>> tf_feat = yeastract |Match(_.target.Each(str.upper), _.feat_name.Each(str.upper))| yeast_feats
    >>> tf_feat  #only showing a few slices...
    Slices: | trans_factor      | target            | sgdid             | feat_type         | feat_qual        
    -----------------------------------------------------------------------------------------------------------
    Type:   | bytes             | bytes             | bytes             | bytes             | bytes            
    Dims:   | yeastract_feats:* | yeastract_feats:* | yeastract_feats:* | yeastract_feats:* | yeastract_feats:*
    Data:   |                   |                   |                   |                   |                  
            | Gcr2              | YAL008w           | S000000006        | ORF               | Verified         
            | Met4              | YAL008w           | S000000006        | ORF               | Verified         
            | Otu1              | YAL008w           | S000000006        | ORF               | Verified
            | ...               | ...               | ...               | ...               | ...


When using a regular ``Match`` operation, any ``target`` row for which no entry can be found in ``feat_name`` will be left out (there are options to prevent this). 



Sidestep: Checking what is linked
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The linking of both datasets is now complete. In this section, we will determine what could be linked, and what not. These steps are performed just to introduce some commands and concepts, and
are not necessary to complete the example.   

First, we do a quick check to determine how many rows in the yeastract dataset could not be matched. A naive approach to this would be::
    
    >>> yeastract.target.Count() - tf_feat.target.Count()
    Slices: | target
    ----------------
    Type:   | int64 
    Dims:   |       
    Data:   |       
            | 72

On a total of 48010 pairs, it appears thus that we lost only a few transcription factor-target pairs. 

This assumes however that `yeast_feats` did not have any non-unique names in `feat_name`, as repeated names will match multiple times to the
same entry in yeastract, and thus increases the number of entries. As an illustration, say we have::

    >>> d1 = Rep([1,2,3,3])
    >>> d2 = Rep([1,3,3])
    >>> d1 |Match| d2
    Slices: | data
    ---------------
    Type:   | int64
    Dims:   | d1:*
    Data:   |
            | 1
            | 3
            | 3
            | 3
            | 3

Thus, two rows with 3's match in ``d1`` match each to two rows of 3's in ``d2``, resulting in 2 * 2 rows of 3's in the output. 

It is easy to determine that `yeast_feats` does not have such non-unique names, using::

    >>> yeast_feats.feat_name[_ != ""].Get(_.Count() == _.Unique().Count())
    Slices: | feat_name
    -------------------
    Type:   | bool     
    Dims:   |          
    Data:   |          
            | True

This command removes the empty feat_names (which do not occur in `yeastract`), and then counts the remaining feat_names, and compares this to a count of the remaining unique feat_names.

However, even a better approach is to circumvent this extra assumption, by checking if the rows in yeastract do actually occur in tf_feat::

    >>> (yeastract |Except| tf_feat.Get(_.trans_factor, _.target)).Count()
    Slices: | trans_factor | target
    -------------------------------
    Type:   | int64        | int64
    Dims:   |              |
    Data:   |              |
            | 72           | 72

This introduces the ``Except`` command. This command ony keeps rows of yeastract that do not occur in tf_feat. These rows are subsequently counted. Note that this gives the same answer as 
we had before. 

A shorter version of this command, that also scales to cases in which `yeastract` has many slices, is the following::
    >>> (yeastract |Except| tf_feat.Get(*yeastract.Names)).Count()

Next, we determine which targets where not matched::

    >>> nonmatched = yeastract.target |Except| tf_feat.target
    >>> nonmatched.Show()
    Slices: | target                       
    ---------------------------------------
    Type:   | bytes                        
    Dims:   | syeastract_syeastract_feats:*
    Data:   |                              
            | YLR157w-c                    
            | A1                           
            | YJL012c-a                    
            | MALT                         
            | MALS                         
            | snR20                        
            | A2                           
            | YAR044w                      
            | RDN5                         
            | YJL017w                      
            | ALD1                         
            | YGR272c                      
            | YBL101w-b                    
            | YBL101w-c                    
            | YDL038c                      
            | YBL101w-a                    
            | TER1                         
            | SUC6                         
            | YDR524w-a                    
            | YDR474c                      
            | YBR075w                      
            | DEX2  

Using ``Except``, we keep only the targets in yeastract that do not occur in ``tf_feat.target``. Another low level way to accomplish the same result
would be::
    
    >>> non_matched = (yeastract.target.Set() - tf_feat.target.Set()).Elem()

``Set`` is used to pack the elements of the (by default last) dimension into a set. A set is a collection of objects
in which each element is unique. That is, adding the string "YLR157W-C" multiple times to a set will result in a set with just one occurence of "YLR157W-C".
Sets have some special operations defined on them. One of them is set substraction, which was used here. It removes all elements in the set of the first operand that
also occur in the set of the second operand, leaving only the elements that do not occur in the second operand. In this case thus the elements that were not matched by the Match operation. 
Next, we use the ``Elem`` operation to unpack the resulting set. 

The names in the list suggest that we might find matching rows by looking either at the ``gene_name`` or ``gene_aliases`` column. 
Before we do this, we first convert each name in nonmatched to uppercase::
    
    >>> nonmatched = nonmatched.Each(str.upper)

First, we check the ``gene_name`` column. This does not give any matches however::
    
    >>> nonmatched |In| yeast_feats.gene_name.Each(str.upper)
    Slices: | result             
    -----------------------------
    Type:   | bool               
    Dims:   | stftargets_sfeats:*
    Data:   |                    
            | False              
            | False              
            | False 
            | ...
    
(Use Show() to see the whole result). This introduces the ``In`` operation, which determines if elements in the left operand occur in the (by default last) dimension of the right operand. 

Next we look at the gene_aliases column. As you might remember this slice does contain nested arrays of aliases. So what will ``|In|`` return here?::

    >>> nonmatched.Each(str.upper) |In| yeast_feats.gene_aliases.Each(str.upper)
    Slices: | result                                    
    ----------------------------------------------------
    Type:   | bool                                      
    Dims:   | stftargets_sfeats:*<feats:*               
    Data:   |                                           
            | [False False False ..., False False False]
            | [False False False ..., False False False]
            | [False False False ..., False False False]
            | ...      

As you can see, ``|In|`` matches with the last dimension of ``gene_aliases``. This means that there are multiple aliases list to be matched, which together with
the multiple names to be tested results in a matrix of results. Of course, this is not what we exactly want. We can solve this using ``Any``::
    
    >>> Any(nonmatched |In| yeast_feats.gene_aliases.Each(str.upper))
    Slices: | result
    -----------------------------
    Type:   | bool
    Dims:   | stftargets_sfeats:*
    Data:   |
            | True
            | True
            | True
            | ...

This aggregates across the ``feats`` dimension, to determine if any of the features had any alias that matched something in our list. As you can see, we indeed found
matches for the targets.

Now that we have found this result, we will use the Match function to find which genes match to these non-matched targets::

    >>> nonmatched_feats = nonmatched |Match(_.target, _.gene_aliases.Each(str.upper))| yeast_feats.Flat()
    >>> nonmatched_feats
    Slices: | target                          | sgdid                           | feat_type                       | feat_qual                       | feat_name                      
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Type:   | bytes[11]                       | bytes                           | bytes                           | bytes                           | bytes[11]                      
    Dims:   | stftargets_sfeats_feats_falias~ | stftargets_sfeats_feats_falias~ | stftargets_sfeats_feats_falias~ | stftargets_sfeats_feats_falias~ | stftargets_sfeats_feats_falias~
    Data:   |                                 |                                 |                                 |                                 |                                
            | YLR157W-C                       | S000028678                      | ORF                             | Uncharacterized                 | YLR157W-E                      
            | YAR044W                         | S000000081                      | ORF                             | Verified                        | YAR042W                        
            | YBL101W-C                       | S000028598                      | ORF                             | Uncharacterized                 | YBL100W-C                      
            | YBL101W-A                       | S000002148                      | transposable_element_gene       |                                 | YBL100W-A                      
            | YJL017W                         | S000003553                      | ORF                             | Uncharacterized                 | YJL016W                        
            | A1                              | S000029660                      | not in systematic sequence of ~ |                                 | MATA1                          
            | YJL012C-A                       | S000003549                      | ORF                             | Verified                        | YJL012C                        
            | MALT                            | S000000502                      | ORF                             | Verified                        | YBR298C                        
            | MALT                            | S000003521                      | ORF                             | Verified                        | YGR289C                        
            | MALT                            | S000029681                      | not in systematic sequence of ~ |                                 | MAL21                          
            | MALT                            | S000029686                      | not in systematic sequence of ~ |                                 | MAL41                          
            | MALT                            | S000029658                      | not in systematic sequence of ~ |                                 | MAL61                          
            | MALS                            | S000000503                      | ORF                             | Verified                        | YBR299W                        
            | MALS                            | S000003524                      | ORF                             | Verified                        | YGR292W                        
            | ...                             | ...                             | ...                             | ...                             | ...    

This shows a possible reason due to which some of these targets do not have an offical name, as a couple of them match to multiple genomic features.

To improve our mapping, we decide to redo our match, and include rows that have a uniuqe ``gene_alias`` match. Our strategy is as follows:

1. Filter out gene_aliases that occur multiple times
2. Convert yeastract targets names that match to the remaining gene_aliases, to the corresponding feat_names
3. Rematch the data. 




First, we determine what names need to be filtered, and filter these from the gene_aliases::
    >>> unique_gene_aliases = yeast_feats.Flat().GroupBy(_.gene_aliases)[Count(_.feat_name) == 1].gene_aliases

    >>> name_alias_list = yeast_feats[_.gene_aliases |In| unique_gene_aliases]

The first command first flattens the nested gene alias lists to get a flat table (If there were would have been more than one nested list 
dimension, we would have had to specify `yeast_feats.Flat(_.gene_aliases)`). 

Next, we group the data on common gene_aliases, and then remove those gene_aliases that have more than more than one associated feat_name. 

Subsequently, we filter the yeast_feats table, such that we only keep the gene_aliases that are in the list of unique gene aliases. 

Next, we convert the yeastract names that occur in the gene_aliases. This can be done using the ``TakeFrom`` command::
    >>> convert_table = name_alias_list.Get(_.gene_aliases.Each(str.upper), _.feat_name).Flat()
    
    >>> yeastract = yeastract.To(_.target, Do=_.Each(str.upper).TakeFrom(convert_table, keep_missing=True))

The TakeFrom command takes a two-slice table (convert_table), and converts the target names that occur in the first slice of the 
table to the names of the second slice of the table.  We set keep_missing to true, to also keep the names that do not occur in the
gene_aliases. 

Now we can redo our match, as we did before::
    >>> tf_feat = yeastract |Match(_.target.Each(str.upper), _.feat_name.Each(str.upper))| yeast_feats


Counting again the number of yeastract rows that could be matched, we find::
    >>> (yeastract |Except| tf_feat.Get(_.trans_factor, _.target)).Count()
    Slices: | trans_factor | target
    -------------------------------
    Type:   | int64        | int64
    Dims:   |              |
    Data:   |              |
            | 6            | 6

Thus, 66 additional rows have been matched. 

Chromosome distribution
~~~~~~~~~~~~~~~~~~~~~~~
First, we save the current dataset. This can be done using::

    >>> Save(tf_feat, 'tf_feat.dat')

The data can be loaded again using::

    >>> tf_feat = Load('tf_feat.dat')


We start with determining for each transcription factor the number of targets per chromosome. To do this, we use a two-dimensional group, grouping both on transcription factor
and chromosome, and counting the number of targets per transcription_factor / chromosome pair::

    >>> tf_feat = tf_feat.GroupBy(_.trans_factor, _.chromosome)
    >>> res = tf_feat.Get(_.trans_factor, _.chromosome, _.target.Count()/"count", _.start).Copy()
    >>> res
    Slices: | trans_factor    | chromosome    | count                                                     | start                                                    
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    Type:   | bytes           | bytes         | int64                                                     | int64?                                                   
    Dims:   | gtrans_factor:* | gchromosome:* | gtrans_factor:*<gchromosome:*                             | gtrans_factor:*<gchromosome:*<gyeastract_feats:~         
    Data:   |                 |               |                                                           |                                                          
            | Gcr2            | 1             | [17 48 60 37 40 32 24 31 80 48 29 52 16 42  8 32]         | [ [136914 36509 2169 186321 21566 31567 222406 221049 92~
            | Met4            | 2             | [ 23  92 100  78  89  61  54  85 165 110  69 110  37  96~ | [ [136914 130799 74020 67520 21566 58462 31567 151166 39~
            | Otu1            | 7             | [ 4 11  9  5  2  1  4  5 11  6  2  4  8  0  0  7]         | [[136914 135665 110430 158619];  [278352 568426 738369 6~
            | Cin5            | 16            | [10 37 54 40 50 38 23 24 67 57 28 65  8 47  9 32]         | [[73518 169375 74020 58462 45022 190193 129270 71786 334~
            | Gcn4            | 10            | [ 22  92 108  92  94  73  49  84 154  97  68 106  24 104~ | [ [169375 36509 192619 130799 67520 203403 58462 54789 4~
            | Zap1            | 11            | [ 4 13 22 10 11 10  2 13 19 17  9 14 12 19  3  7]         | [[186321 151166 99697 155005];  [724456 686901 382030 81~
            | Yap7            | 9             | [ 2 13 11 12 14 15  2 11 30 19 15 12  4 13  3  8]         | [[130799 119541];  [724456 168423 583720 582652 331511 7~
            | Ste12           | 14            | [ 37 153 215 158 129 113  93 131 227 200  93 182  70 159~ | [ [169375 164187 92270 192619 67520 21566 69525 203403 1~
            | Arr1            | 4             | [ 4  1 85 87 51 54 28 76  4 90 52 97  6 81 26  1]         | [[31567 222406 120225 119541] [316968];  [828625 87981 4~
            | Aft2            | 12            | [ 1 15 20 11  9  8  6 14 22 25  9 18  2 14  6 13]         | [[110430];  [382030 633622 447709 635146 393123 426489 5~
            | Ecm22           | 8             | [ 6 15 25 33 22 13 13 17 31 27 13 17  4 11  6 17]         | [[94687 218140 2169 186321 220198 177023];  [444693 7144~
            | Ino4            | 15            | [ 7 40 69 49 38 22 26 53 86 45 43 61 24 40  7 27]         | [[94687 21566 69525 45022 71786 13743 45899];  [13879 31~
            | Aft1            | 3             | [ 22  75 114  82  72  53  50  71 127  87  65 104  26  94~ | [ [36509 87031 129019 203403 31567 222406 45022 190193 1~
            | Sok2            | 13            | [ 31  62  98  68  62  51  54  69  97  58  70 109  24  72~ | [ [218140 164187 92270 82706 192619 67520 21566 175135 6~
            | ...             | ...           | ...                                                       | ...      


Note that each slice has now a different dimension. Trans_factor and chromosome both have a single dimension, with all unique values. The count slice contains a matrix, 
with counts for each transcription_factor/chromosome pair, and ``start`` contains for each transcription factor/chromosome pair a list of all gene start positions.

To calculate now a correlation correlation between transcription factors, based on if they target the same chromosomes, we can simply do::

    >>> Corr(res.count)

However, the resulting correlations are positively biased as we did not control for the different numbers of genes on each chromosome.
Therefore, we normalize the count data first by dividing by the total number of targets per chromosome::
    
    >>> normchrom_counts = res.count.Cast("real64") / res.count.Sum("gtrans_factor")
    >>> Corr(normchrom_counts)
    Slices: | count                                                                                                                                                  
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    Type:   | real64                                                                                                                                                 
    Dims:   | gtrans_factor:*<gtrans_factor:*                                                                                                                        
    Data:   |                                                                                                                                                        
            | [ 1.          0.84058821  0.56884259  0.46862953  0.75701405  0.60343542;   0.54598067  0.78050323  0.1699565   0.39857328  0.695562    0.63384689;   ~
            | [ 0.84058821  1.          0.34636467  0.40345917  0.86052624  0.61812576;   0.68426206  0.92512018  0.45395797  0.67508341  0.72107807  0.78040341;   ~
            | [ 0.56884259  0.34636467  1.         -0.15274769  0.07486133  0.58630083;  -0.02208724  0.38574002 -0.41424377 -0.10392024  0.22757454  0.38601169;   ~
            | [ 0.46862953  0.40345917 -0.15274769  1.          0.71668069  0.07167468;   0.42931201  0.35479816  0.32588663  0.34150679  0.43178954  0.21771547;   ~
            | [ 0.75701405  0.86052624  0.07486133  0.71668069  1.          0.40977524;   0.70526491  0.78290387  0.52340019  0.64076449  0.73000811  0.62646315;   ~
            | [ 0.60343542  0.61812576  0.58630083  0.07167468  0.40977524  1.;   0.48239639  0.64139371  0.21699255  0.25563464  0.22584824  0.6321228;   0.5578196~
            | [ 0.54598067  0.68426206 -0.02208724  0.42931201  0.70526491  0.48239639;   1.          0.54775513  0.41362316  0.54245825  0.47982939  0.57266579;   ~
            | [ 0.78050323  0.92512018  0.38574002  0.35479816  0.78290387  0.64139371;   0.54775513  1.          0.48832401  0.73255135  0.74640988  0.70272516;   ~
            | [ 0.1699565   0.45395797 -0.41424377  0.32588663  0.52340019  0.21699255;   0.41362316  0.48832401  1.          0.54753551  0.45464972  0.41855642;   ~
            | [ 0.39857328  0.67508341 -0.10392024  0.34150679  0.64076449  0.25563464;   0.54245825  0.73255135  0.54753551  1.          0.53611385  0.49655451;   ~
            | [ 0.695562    0.72107807  0.22757454  0.43178954  0.73000811  0.22584824;   0.47982939  0.74640988  0.45464972  0.53611385  1.          0.54252082;   ~
            | [ 0.63384689  0.78040341  0.38601169  0.21771547  0.62646315  0.6321228;   0.57266579  0.70272516  0.41855642  0.49655451  0.54252082  1.;   0.7725603~
            | [ 0.84629762  0.94229773  0.2511703   0.4532385   0.8736701   0.55781963;   0.6038615   0.89819293  0.58601108  0.64679905  0.75517794  0.77256031;   ~
            | [ 0.73136577  0.58847393  0.41657058  0.49813967  0.64892351  0.26202592;   0.24997758  0.5509231   0.13508773  0.25581194  0.58083685  0.46142624;   ~
            | ...                                  

Note that we first cast to double, as integer division will only result in whole integers. We sum along the `gtrans_factor` dimension to determine the number of targets
per chromosome. The division operator knows on which dimension it should divide and how it should broadcast, as it can use the dimension identities. 





As you can see, a square matrix is calculated with all correlation coefficients. What if we now want to calculate a correlation between chromosomes instead?::
    >>> normchrom_counts = res.count.Cast("real64") / res.count.Sum("gtrans_factor")
    >>> Corr(res.count.Transpose())
    Slices: | count                                                                                                                                                                    
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Type:   | real64                                                                                                                                                                   
    Dims:   | gchromosome:*<gchromosome:*                                                                                                                                              
    Data:   |                                                                                                                                                                          
            | [ 1.          0.89794002  0.9010322   0.88750552  0.88835388  0.87571585;   0.90482509  0.87475769  0.89191886  0.86099942  0.8941382   0.89517086;   0.0478764   0.8541~
            | [ 0.89794002  1.          0.9522306   0.93640212  0.93131438  0.94136578;   0.91872408  0.93151153  0.97403148  0.9407366   0.91162125  0.9375463;   0.09015854  0.91941~
            | [ 0.9010322   0.9522306   1.          0.97481982  0.95779989  0.96366473;   0.94670432  0.96283067  0.94969844  0.96663783  0.94634064  0.9705861;   0.07341518  0.91507~
            | [ 0.88750552  0.93640212  0.97481982  1.          0.94343915  0.95542594;   0.93340581  0.96223144  0.94080503  0.95758314  0.94135692  0.96550212;   0.07373773  0.8979~
            | [ 0.88835388  0.93131438  0.95779989  0.94343915  1.          0.95198183;   0.94894526  0.95801655  0.94905974  0.94857988  0.94038115  0.95604521;   0.05784699  0.8935~
            | [ 0.87571585  0.94136578  0.96366473  0.95542594  0.95198183  1.;   0.92715493  0.95535543  0.94584436  0.95664342  0.92404194  0.94985628;   0.09451878  0.89830133  0.~
            | [ 0.90482509  0.91872408  0.94670432  0.93340581  0.94894526  0.92715493;   1.          0.93897798  0.92797838  0.91586707  0.92713411  0.95517812;   0.06729443  0.8927~
            | [ 0.87475769  0.93151153  0.96283067  0.96223144  0.95801655  0.95535543;   0.93897798  1.          0.94242741  0.95495536  0.94368535  0.96420902;   0.0483322   0.8920~
            | [ 0.89191886  0.97403148  0.94969844  0.94080503  0.94905974  0.94584436;   0.92797838  0.94242741  1.          0.95004305  0.92081038  0.94154646;   0.07403729  0.9210~
            | [ 0.86099942  0.9407366   0.96663783  0.95758314  0.94857988  0.95664342;   0.91586707  0.95495536  0.95004305  1.          0.92405937  0.94925448;   0.07162337  0.9030~
            | [ 0.8941382   0.91162125  0.94634064  0.94135692  0.94038115  0.92404194;   0.92713411  0.94368535  0.92081038  0.92405937  1.          0.94417153;   0.05994558  0.8858~
            | [ 0.89517086  0.9375463   0.9705861   0.96550212  0.95604521  0.94985628;   0.95517812  0.96420902  0.94154646  0.94925448  0.94417153  1.;   0.07112272  0.88703263  0.~
            | [ 0.0478764   0.09015854  0.07341518  0.07373773  0.05784699  0.09451878;   0.06729443  0.0483322   0.07403729  0.07162337  0.05994558  0.07112272;   1.          0.0832~
            | [ 0.85419801  0.91941182  0.91507757  0.89795194  0.89359741  0.89830133;   0.8927235   0.89203255  0.92109235  0.90301833  0.88589314  0.88703263;   0.08322617  1.    ~

For this we use the ``Transpose`` operation, which can be used to reorder the dimensions of slices. Of course, from this matrix it is hard to identify which columns/rows correspond to which chromosome.
So we would like to order on chromosome number. As it is currently a bytes type, the ``Sort`` operation would perform an alphabetic ordering which is not what we want. So, we cast it to an integer type::
    
    >>> res = res.To(_.chromosome, Do=_.Cast("int?"))

Next, we ``Sort`` the data on chromosome number, and then calculate the correlation, showing both chromosome number and correlation slice::

    >>> res.Sort(_.chromosome).Get(_.chromosome, Corr(_.count.Transpose()/"chromo_corr")).Show()
    Slices: | chromosome    | chromo_corr                                                                                                                                              
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Type:   | int32?        | real64?                                                                                                                                                  
    Dims:   | gchromosome:* | gchromosome:*<gchromosome:*                                                                                                                              
    Data:   |               |                                                                                                                                                          
            | 1             | [1.0 0.897940020734 0.854198008099 0.89191885686 0.9054486057;  0.847700761107 0.901032202462 0.894138196259 0.904825094876 0.888353880838;  0.875715852~
            | 2             | [0.897940020734 1.0 0.919411822605 0.974031484327 0.921561322359;  0.903932265285 0.952230596519 0.911621246046 0.918724075112 0.93131438346;  0.9413657~
            | 3             | [0.854198008099 0.919411822605 1.0 0.921092346661 0.887224003105;  0.882968908511 0.915077574905 0.885893140026 0.892723501095 0.893597408111;  0.898301~
            | 4             | [0.89191885686 0.974031484327 0.921092346661 1.0 0.932660005081;  0.899203041744 0.949698439024 0.920810377233 0.927978380412 0.949059735695;  0.9458443~
            | 5             | [0.9054486057 0.921561322359 0.887224003105 0.932660005081 1.0;  0.884149425526 0.930355325662 0.890784666788 0.934683503715 0.940616915228;  0.91522908~
            | 6             | [0.847700761107 0.903932265285 0.882968908511 0.899203041744 0.884149425526;  1.0 0.9341103835 0.916102953444 0.902049890891 0.937622750781;  0.92873987~
            | 7             | [0.901032202462 0.952230596519 0.915077574905 0.949698439024 0.930355325662;  0.9341103835 1.0 0.946340639279 0.946704322698 0.957799890746;  0.96366472~
            | 8             | [0.894138196259 0.911621246046 0.885893140026 0.920810377233 0.890784666788;  0.916102953444 0.946340639279 1.0 0.927134108999 0.940381149843;  0.924041~
            | 9             | [0.904825094876 0.918724075112 0.892723501095 0.927978380412 0.934683503715;  0.902049890891 0.946704322698 0.927134108999 1.0 0.94894526088;  0.9271549~
            | 10            | [0.888353880838 0.93131438346 0.893597408111 0.949059735695 0.940616915228;  0.937622750781 0.957799890746 0.940381149843 0.94894526088 1.0;  0.95198183~
            | 11            | [0.875715852477 0.941365783875 0.898301329812 0.945844355986 0.915229084942;  0.928739873918 0.963664725752 0.924041936219 0.927154931247 0.9519818303; ~
            | 12            | [0.860999420572 0.940736601384 0.903018327462 0.950043051223 0.898309825619;  0.931077837874 0.96663782684 0.92405937455 0.915867066669 0.948579877392; ~
            | 13            | [0.890483734871 0.947971265947 0.89653077661 0.949365125369 0.925409063547;  0.92909231685 0.975192117256 0.935007023899 0.950230753511 0.959563025937; ~
            | 14            | [0.874757692555 0.93151152716 0.892032546886 0.942427414978 0.906462858776;  0.928214562702 0.962830665329 0.943685349899 0.938977976517 0.958016551185;~
            | 15            | [0.895170862494 0.937546297182 0.887032632031 0.941546463876 0.926728792948;  0.913203658988 0.970586099908 0.944171525867 0.955178119088 0.956045211742~
            | 16            | [0.887505518635 0.936402117242 0.897951938723 0.940805029861 0.905165349958;  0.920197445731 0.974819815867 0.94135692062 0.93340580682 0.943439150747; ~
            | 17            | [0.320441396715 0.434059532382 0.39111506662 0.374280121929 0.346409832854;  0.385450229121 0.365643846425 0.371215994602 0.312121467777 0.308091894183;~
            | --            | [0.0478763953923 0.09015853553 0.0832261725262 0.0740372911537;  0.0609880189341 0.132869014898 0.0734151794607 0.0599455812233;  0.067294433364 0.05784~

We see that chromosome number 17 has a relatively low correlation. Is this due to a low number of targets on this chromosome?::

    >>> res.Get(_.chromosome, _.count.Sum("gtrans_factor")).Show()
    Slices: | chromosome    | count
    ---------------------------------------
    Type:   | int32?        | int32
    Dims:   | gchromosome:* | gchromosome:*
    Data:   |               |
            | 1             | 834
            | 2             | 3274
            | 7             | 4300
            | 16            | 3512
            | 10            | 2990
            | 11            | 2491
            | 9             | 1901
            | 14            | 3025
            | 4             | 6014
            | 12            | 4135
            | 8             | 2402
            | 15            | 4341
            | --            | 11
            | 3             | 1442
            | 13            | 3624
            | 6             | 1034
            | 5             | 2672
            | 17            | 73

Indeed it seems that the low number of targets is the cause. Note that we give ``Sum`` the dimension 
accross which it has to sum the results, as normally it would take the last dimension, and calculate a 
Sum for each transcription factor, which is not what we want. 

As last step, we like to calculate to what extent transcription factors target specific chromosomes. 

Our first approach calculates this using::

    >>> res.count.Sort().Sum("gtrans_factor")
    Slices: | count
    -----------------------
    Type:   | int32
    Dims:   | gchromosome:*
    Data:   |
            | 1
            | 72
            | 634
            | 948
            | 1279
            | 1657
            | 1986
            | 2281
            | 2546
            | 2790
            | 3020
            | 3292
            | 3551
            | 3813
            | ...

That is, we sort the counts for each transcription factor, and then sum the most visited chromosome for each transcription factor,
the second most visited, and so on. 

However, this does not control for the fact that some chromosomes have much more targets than others. So, Now we can finish::

    >>> normalized_counts.Sort().Sum("gtrans_factor").Show()
    Slices: | count          
    -------------------------
    Type:   | real64         
    Dims:   | gchromosome:*  
    Data:   |                
            | 0.0299760191847
            | 0.204644460173 
            | 0.556406790278 
            | 0.65490950643  
            | 0.718375717458 
            | 0.777312026173 
            | 0.83069212706  
            | 0.886076633193 
            | 0.928012809055 
            | 0.962799237777 
            | 1.00445068188  
            | 1.05156749621  
            | 1.10260676947  
            | 1.16260229285  
            | 1.22164191558  
            | 1.30508535495  
            | 1.47473492256  
            | 3.12810523972  
    
It seems that indeed there is some chromosome specificness for transcription factors
(although making this a hard conclusion would probably require a permutation analysis). 

We plot the results using matplotlib::

    >>> from matplotlib.pylab import *
    >>> plot(normalized_counts.Sort().Sum("gtrans_factor")())
    >>> title("Chromosome specificness of transcription factors")
    >>> ylabel("Normalized target counts")
    >>> xlabel("Less visited --> Most visited chromosome")
    >>> show()



.. image:: chromo_spec.png


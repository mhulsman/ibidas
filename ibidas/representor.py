import sys
import copy
import itertools
from collections import defaultdict
import numpy

from query_graph import Node
from itypes import rtypes
from constants import *
from thirdparty import tableprint, console

_delay_import_(globals(),"utils","util","context","infix")
_delay_import_(globals(),"itypes", "dimensions","dimpaths")
_delay_import_(globals(),"wrappers","python","wrapper")
_delay_import_(globals(),"query_context")
_delay_import_(globals(),"engines")
_delay_import_(globals(),"repops")
_delay_import_(globals(),"repops_multi")
_delay_import_(globals(),"repops_dim")
_delay_import_(globals(),"repops_slice")
_delay_import_(globals(),"repops_funcs")
_delay_import_(globals(),"ops")

class Representor(Node):
    """Representor is the primary object in Ibidas. It represents
    a data set, accesible through slices.

    Properties can be accessed to obtain information about this object:
    
        * Names: Slice names

        * Type: Data type

        * Slices: List of slices

        * I: Info on slices/types/dims without executing the query

        * Depth: Maximum number of dimensions in slices

    Slices can be accessed as attributes, e.g: obj.slicename

    Note that all slice names should follow the python syntax rules for variable
    names, AND use only lower case letters (to distinguish them from method names, 
    which start all with an uppercase letter). 

    Special attribute access can be obtained through so-called axis specifiers:

        * Bookmarks:  obj.Bbookmarkname   Access set of slices with certain bookmark (see Bookmark method)
        
        * Dimensions: obj.Ddimname        Access all slices with a certain dimension

        * Elements:   obj.E[dimname]      Access Elements of packed arrays. Optional dimname specifies which dimensions to unpack.
                                          Slices without that dimension as outermost dimension are not unpacked. 

        * Fields:     obj.Ffieldname      Access Fields of packed tuples. obj should have only one slice. 

        * Left/Right: obj.L, obj.R        Special nested bookmarks set by e.g. Match operation to allow 
                                          backtracking to separate sources. eg::
                                          
                                          >>> ((x |Match| y) |Match| z).LR  
                                          
                                          gives all slices of y (first go Left (get xy), then R (get y)). 


    Representor objects can be created from python data objects using the 'Rep' function, e.g.::

        >>> Rep([('a',3),('b',4)])

    """

    _state = 0                 #default value
    _slices = []               #default value

    def _initialize(self, slices):
        assert isinstance(slices, tuple), "slices should be a tuple"
        self._slices = slices

    def _typesKnown(self):
        return isinstance(self._slices, tuple) and not any([slice.type == rtypes.unknown for slice in self._slices])

    def _slicesKnown(self):
        return isinstance(self._slices,tuple)

    def _checkState(self,check_type=False):
        if(check_type is True):
            found = self._typesKnown()
        else:
            found = self._slicesKnown()

        if(not found):
            self._initialize(self._getResultSlices(endpoint=False))

    def Show(self,table_length=100):
        """Prints table of contents.

        :param table_length: Number of rows to show for each dimension (default: 100)

        Show can be used to view a larger part of the table then the default output (15 rows)
        you get in Ipython/Ibidas or by using str(). 

        Show returns its representor object, allowing you to include it at any point in a query
        to observe results, e.g.::

        >>> x.Unique().Show() |Match| y 

        """
        nself = repops.NoOp(self)
        nself._table_length = table_length
        return nself

    def __str__(self, print_data=True, table_length=None):
        self._checkState()

        if table_length is None:
            if '_table_length' in self.__dict__:
                table_length = self._table_length
            else:
                table_length = 15
        
        names = ["Slices:"] + [s.name for s in self._slices]
        types = ["Type:"] + [str(s.type) for s in self._slices]
        longdims = ["Dims:"]
        for s in self._slices:
            dim_str = []
            for  dim in s.dims:
                dim_str.append(str(dim))
            dim_str = "<".join(dim_str)
            longdims.append(dim_str)
                
        dims = ["Dims:"]
        last_dim = ()
        for s in self._slices:
            dim_str = []
            for pos, dim in enumerate(s.dims):
                if len(last_dim) > pos and dim == last_dim[pos]:
                    dim_str.append(".")
                else:
                    dim_str.append(str(dim))
            last_dim = s.dims
            dim_str = "<".join(dim_str)
            dims.append(dim_str)
        
        rows = [names, types, longdims]
       
        opath = dimpaths.getOrderDim([s.dims for s in self._slices])


        if(print_data):
            first_dims = set([s.dims[0] for s in self._slices if s.dims])
            if(first_dims):
                for dim in first_dims:
                    self = self.Filter(slice(None,table_length + 1),dim=dim)
                data = self()
            else:
                data = self()
            if(len(self._slices) == 1):
                data = (data,)

            cols = [[""] * (table_length + 1)]
            maxlen = 0
            
            maxwidth = [0]
            for dcol,s in zip(data, self._slices):
                if(len(s.dims) >= 1):
                    col = [str(row).replace('\n','; ') for row in dcol[:table_length]]
                    maxlen = max(len(col),maxlen)
                    if(len(dcol) > table_length):
                        col[table_length-1] = "..."
                    elif(len(col) < table_length):
                        col.extend([""] * (table_length - len(col)))
                        
                else:
                    col = [""] * (table_length + 1)
                    col[0] = str(dcol)
                    maxlen = max(1,maxlen)
                cols.append(col)
            
            rows.append([""] * len(rows[0]))
            rows[-1][0] = "Data:"
            for i in range(maxlen):
                row = [col[i] for col in cols]
                rows.append(row)
        
        console_height, console_width = console.getTerminalSize()
        widths = tableprint.calc_width(rows)
        indices = tableprint.advise_splits(console_width, widths, 3)

        res = ""
        for i in range(len(indices)):
            srows = tableprint.select_cols(rows, indices[i])
            if(i > 0):
                res += "\n\n"
                col0 = tableprint.select_cols(rows,0)
                srows = tableprint.prepend_col(srows,col0)
            widths = tableprint.calc_width(srows)
            nwidths = tableprint.optimize_width(console_width, widths, 3)
            srows = tableprint.row_crop(nwidths, srows)
            res += tableprint.indent(srows,hasHeader=True) 
        
        if opath:
            dim_str = []
            for  dim in opath:
                dim_str.append(str(dim))
            dim_str = "<".join(dim_str)
            res = res + '\nDim order: ' + dim_str 

        return res 

    def _getInfo(self):
        class Info(object):
            def __repr__(self):
                return self.info
        k = Info()
        k.info = self.__str__(False)
        return k
    I=property(fget=_getInfo)
    Info=I

    #def __repr__(self):
    #    return str(self.__class__)
    __repr__ = __str__

    def _axisF(self, name): 
        if(name == ""):
            name = None
        return repops_slice.UnpackTuple(self, name)

    def _axisE(self, name):
        if(name == ""):
            name = None
        return repops_dim.UnpackArray(self, name)

    def _axisD(self, name):
        #slices = self._active_dim_slice_dict[name] 
        return repops_slice.ProjectDim(self, name)

    def _axisB(self, name):
        return repops_slice.ProjectBookmark(self,name)

    def _axisL(self, name):
        bname = "!L"
        while(name and (name[0] == "L" or name[0] == "R")):
            bname = bname + name[0]
            name = name[1:]

        r = repops_slice.ProjectBookmark(self,bname)
        if(name):
            r = self.__getattr__(name)
        return r
        
    def _axisR(self, name):
        bname = "!R"
        while(name and (name[0] == "L" or name[0] == "R")):
            bname = bname + name[0]
            name = name[1:]

        r = repops_slice.ProjectBookmark(self,bname)
        if(name):
            r = self.__getattr__(name)
        return r

    def __getattr__(self, name):
        if(not name):
            return self

        axis_letter = name[0]
        #should be upper case
        if(not axis_letter.upper() == axis_letter or axis_letter == "_"):
            try:
                return repops_slice.Project(self,name)
            except AttributeError, error:
                #reraise attribute error as runtime error, 
                #so that python will not
                #attempt to find attribute in another way
                #(thereby raising a new exception)
                exc_class, exc, traceback = sys.exc_info()
                raise RuntimeError, RuntimeError(error.message), traceback
        elif(hasattr(self, '_axis' + axis_letter)):
            try:
                name = name[1:]
                return getattr(self, "_axis" + axis_letter)(name)
            except AttributeError, error:
                #reraise attribute error as runtime error, 
                #so that python will not
                #attempt to find attribute in another way
                #(thereby raising a new exception)
                exc_class, exc, traceback = sys.exc_info()
                raise RuntimeError, RuntimeError(error.message + " (and there is also no method '" + axis_letter + name + "')"), traceback
        else:
            raise RuntimeError, 'No method with name: ' + name

    def _getAttributeNames(self):
        if not self._slicesKnown():
            return []
        else:
            #print ""
            #print self.__str__(False)
            return [slice.name for slice in self._slices]

    def Copy(self, log=False, debug=False):
        """Executes the current query.

        Normally, query operations (e.g. obj + 3) are not executed immediately. Instead
        these operations are performed simultaneously when output is requested. This allows
        us to optimize these operations all together, or e.g. translate them into a SQL
        query. 

        However, this behaviour is not always what is needed, e.g::

            >>> x = very expensive query
            >>> print x[10:20]
            >>> print x[10:30]
            
        would execute the query saved in x two times (as ibidas due to being part of an 
        interpreted language cannot analyze the whole script to determine that the output is 
        required twice). 

        To prevent this, one can instead do::

            >>> x = (very expensive query).Copy()

        executing the expensive part of the only query once. 


        :param log:  Setting this to true will print the amount of time that is spent in any of the passes of
                     they query optimizer (default: False)

        :param debug: Setting this to true will output the query tree before optimization and after processing, through XML-RPC 
                      for visualization in Cytoscape. This requires that Cytoscape is running, with an activated XML-RPC plugin 
                      listening at port 9000. 
                     

        """

        res = wrapper.SourceRepresentor()
        res._initialize(tuple(self._getResultSlices(endpoint=False, log=log, debug=debug)))
        return res

   
    def __reduce__(self):
        self = self.Copy()
        slices = self._getResultSlices(endpoint=False)
        for slice in slices:
            slice.source = None
        return (python.PyRepresentor, (slices,))

    def __copy__(self):
        nself = Representor()
        nself.__class__ = self.__class__
        nself.__dict__ = self.__dict__.copy()
        return nself

   
    def _getType(self):
        """Returns type of this object.
           If multiple slices, returns tuple type of slice types.
        """
        if(len(self._slices) == 1):
            return self._slices[0].type
        else:
            return rtypes.TypeTuple(False, 
                    tuple([slice.type for slice in self._slices]), 
                    tuple([slice.name for slice in self._slices]))
    Type=property(fget=_getType)

    def _getSlices(self):
        self._checkState()
        return list(self._slices)
    Slices=property(fget=_getSlices)

    def _getDepth(self):
        """Returns max dimension depth (number of dimensins) of
           slices in this representor. 
        """
        return max([len(slice.dims) for slice in self._slices])
    Depth=property(fget=_getDepth)
    

    def _getNames(self):
        """Returns names of all slices"""
        self._checkState()
        return [slice.name for slice in self._slices]
    Names=property(fget=_getNames)


    def _getDims(self):
        """Returns dims, ordered according to the order shown below a dataset printout. 

           Note that dimensions that occur multiple times in the same slice will be repeated (if this is not what is needed, use DimsUnique).
        
        """
        opath = dimpaths.getOrderDim([s.dims for s in self._slices])
        return opath
    Dims=property(fget=_getDims)
    
    def _getDimsUnique(self):
        """Returns list of unique dims, ordered as used by DimRename.
        """
        unique_dimpath = util.unique(sum([slice.dims for slice in self._slices],dimpaths.DimPath())) 
        return unique_dimpath
    DimsUnique=property(fget=_getDimsUnique)

    def __getitem__(self, condition):
        self._checkState()
        if(not isinstance(condition, tuple)):
            condition = (condition,)
        
        #add dimensions first
        ipos = 0
        for pos, cond in enumerate(condition):
            if(isinstance(cond,NewDim)):
                self = repops_dim.InsertDim(self,pos + ipos, cond.name)
            elif(cond is Ellipsis):
                ulength = len(dimpaths.uniqueDimPath([s.dims for s in self._slices], only_unique=True))
                rem_length = len([c for c in condition[(pos + 1):] if not isinstance(cond,NewDim)])
                newnextpos = ulength - rem_length
                curnextpos = pos + ipos
                ipos += newnextpos - curnextpos #skip some dims
        
        #next, perform filtering in backwards order
        #note: cannot use Ellipsis in condition, will perform Ellipsis == elem,
        #which wille xecute as query and fail/be slow
        if(any([elem is Ellipsis for elem in condition])):  
            ncond = len(dimpaths.getOrderDim([s.dims for s in self._slices]))
        else:
            ncond = len(condition)
        
        for pos, cond in enumerate(condition[::-1]):
            if(isinstance(cond, context.Context)):
                cond = context._apply(cond, self)
            if(isinstance(cond,slice) and cond.start is None and 
                                                cond.stop is None and 
                                                cond.step is None):
                pass
            elif(cond is Ellipsis):
                ncond = len(condition)
            elif(isinstance(cond,NewDim)):
                pass
            elif(len(condition) == 1):
                self = repops_multi.Filter(self, cond)
            else:
                self = repops_multi.Filter(self, cond, ncond - pos - 1)
        return self
     
    def Redim(self, *args, **kwds):
        """Assign new dimensions

        example: .Redim('new_dimname', _.f0)
        Assign new dim with name 'new_dimname' to first
        dimension of slice f0

        example: .Redim('new_dimname', f0=1)
        Assign new dim with name 'new_dimname' to second
        dimension of slice f0

        example: .Redim('new_dimname', {_.Dd1.Without('f0'):1, 'f0':1})
        Assign all slices with dimension d1 (except f0) as first dim
        a new dim with name 'new_dimname'. Do the same to slice f0, 
        but as second dimension. 
        """
        return repops_dim.Redim(self, *args, **kwds)
          
    def Filter(self, condition, dim=LASTCOMMONDIM, mode=None):
        """Performs filtering on this dataset using ``condition``.
           
           :param condition: condition to filter on

                * condition should have only a single slice.

                Various data types can be used:

                * Bool: last dim of condition should be equal to a dim in this representor. Filtering occurs on the matching dim. 

                * Integer: selects element from a dimension (see below how this is specified). Collapses the dimension it is applied on. 

                * Array (of integers): selects positions from a dimension indicated by integers in array.

                * Slice: selects slice of elements from dimension (note that we refer here to the Python slice object, e.g. slice(0,3), not the Ibidas slice concept). 

           :param dim: Dim to apply the filtering on. 

                * If no dim given, filtering is performed on the last common dimension of the slices (except for bool types, where the dimension of the condition specifies the filtered dimension).

                * Integer: identifies dimension according to dim order (printed at the end of a representor printout)

                * Long: identifies dimension according to common dimensions shared by all slices (default: -1L)

                * String: dimension name

                * Dim object: x.Slices[0].dims[2]

                * Dimpath object: e.g. x.Slices[0].dims[:2]

           :param mode: Determines broadcasting method. 

                * "pos"  Postion-based broadcasting ('numpy'-like broadcasting), not based on dimension identity. 

                * "dim"  Dimension-identity based broadcasting (normal 'ibidas' broadcasting)

                * None:  (Default). Determined based on input. Non-representation objects use position-based broadcasting
                         Representation objects by default use dimension-based, except if they are prepended by a '+' operator, 
                         e.g::

                         >>> x.Filter(+conditionrep)

           What is done to dimensions in the conditions that are not in the data source? Here we follow
           the default rules in Ibidas for broadcasting. 

               * First, the dimension in the source that is going to be filtered is identified (see previous sections)
               
               * Secondly, we match this dimension to the last dimension in the condition. 
               
               * All remaining dimensions are broadcasted against each other.

           The examples use the following dataset::

               >>> x = Rep([[1,2,3],[4,5,6]]) 
               Slices: | data     
               -------------------
               Type:   | int64    
               Dims:   | d1:2<d2:3
               Data:   |          
                       | [1 2 3]  
                       | [4 5 6]  
               
               Dim order: d1:2<d2:3

           * Example: integer filtering
           
             Filtering the first element::   
             
                 >>> x.Filter(0) 
                 Slices: | data
                 ---------------
                 Type:   | int64
                 Dims:   | d1:2
                 Data:   |
                         | 1
                         | 4
                 
                 Dim order: d1:2

             This example matches the last common dimension (d2), and selects the first element.
             This collapses dimension d2.

             Note that if no special keywords are required, one can also use brackets to specify the filter operation::

                 >>> x[0]

             is equivalent to the previous filtering operation.  

             Using the Filter command, we can however also specify that we want to filter a specific dimension::

                 >>> x.Filter(0, dim='d1')
                 Slices: | data 
                 ---------------
                 Type:   | int64
                 Dims:   | d2:3 
                 Data:   |      
                         | 1    
                         | 2    
                         | 3    

                 Dim order: d2:3

             One can also use positional indices for the dimension (according to Dim order in the printout)::
                 
                 >>> x.Filter(0, dim=0)
           

           * Example: Boolean filtering
 
             Filtering on boolean constraints::

                 >>> x.Filter(_ > 2)
                 Slices: | data      
                 --------------------
                 Type:   | int64     
                 Dims:   | d1:2<fd2:~
                 Data:   |           
                         | [3]       
                         | [4 5 6]   

                 Dim order: d1:2<fd2:~

             Here, the _ operator refers to the enclosing scope, i.e. 'x'.  Equivalent is::
                 
                 >>> x[_ > 2]

           * Example: Slice filtering
             
             One can also filter using  Python slices::
             
                 >>> x.Filter(slice(0,2))

                 Slices: | data      
                 --------------------
                 Type:   | int64     
                 Dims:   | d1:2<fd2:*
                 Data:   |           
                         | [1 2]     
                         | [4 5]     

                 Dim order: d1:2<fd2:*

             Note that this is equivalent to::
                 
                 >>> x[0:2]

             (here we do not have to explicitly construct the slice object, as python accepts for this the x:y syntax. Unfortunately, this syntax is not allowed outside brackets).

           * Integer filtering (with arrays)
           
             Filtering on array::

                 >>> x.Filter([0,1])
                 Slices: | data
                 ---------------
                 Type:   | int64
                 Dims:   | d1:2
                 Data:   |
                         | 1
                         | 5

                 Dim order: d1:2
                 
             This is maybe not what most expected. Note that the filtering is applied on dimension 'd2'. The dimension of the [0,1] array is mapped
             to dimension 'd1'. Thus, from the first position along 'd1' (first row), we select the 0th element from dim d2,
             and from the second position along 'd1', we select the 1th element along dim d2. 

             We used here positional broadcasting, as the input was not an Ibidas object. That is, the dimension of [0,1] was mapped to the dimension 'd1', even though
             these do not have the same identity. We can however also specify that we want to do identity based broadcasting::

                     >>> x.Filter([0,1],mode='dim')

                     Slices: | data            
                     --------------------------
                     Type:   | int64           
                     Dims:   | d1:2<d3:2
                     Data:   |                 
                             | [1 2]           
                             | [4 5]           

                     Dim order: d1:2<d3:2

             
             This applies the [0,1] array as filter on the d2 dimension, transforming it into dimension d3. 

             What actually happens is slightly more complicated however: the integers in the [0,1] list are mapped as filters to dimension 'd2'. 
             This filtering is however done for each element in the [0,1] list, which has dimension 'd3'. As this dimension is not equal to dimension 'd1', it is broadcasted: virtually, 
             the dataset is converted into one with dimensionds d1:2<d3:2<d2:2. Upon applying the filter, dimension 'd2' collapses, resulting in a dataset with dimension  'd1:2<d3:2'. 
             Of course, in practice, we optimize this broadcasting step away. 

             Such broadcasting can also happen when using position-based broadcasting, e.g.::

                     >>> x.Filter([[0,1],[0,2]])
                     
                     Slices: | data
                     -------------------
                     Type:   | int64
                     Dims:   | d3:2<d1:2
                     Data:   |
                             | [1 5]
                             | [1 6]

                     Dim order: d3:2<d1:2

             First, we do the same positional broadcasting, filtering dimension 'd2', and mapping the second dimension of [[0,1],[0,2]] to dimension 'd1'. But then we are left with the
             extra first dimension of [[0,1],[0,2]], which is called 'd3'. This dimension is next broadcasted. As 'd1' is already mapped to, the dimension is put in front of 'd1'.

             We can make this quite complicated, e.g.::
                 
                 x.Filter([[0,1],[0,2,1]],mode='dim')

                 Slices: | data           
                 -------------------------
                 Type:   | int64          
                 Dims:   | d1:2<d4:2<d3:~ 
                 Data:   |                
                         | [[1 2] [1 3 2]]
                         | [[4 5] [4 6 5]]

                 Dim order: d1:2<d4:2<d3:~

             or::

                 x.Filter([[0,1],[0,1,1]],mode='dim',dim='d1')

                 Slices: | data
                 ---------------------------------------
                 Type:   | int64
                 Dims:   | d4:2<d3:~<d2:3
                 Data:   |
                         | [[1 2 3];  [4 5 6]]
                         | [[1 2 3];  [4 5 6];  [4 5 6]]

                 Dim order: d4:2<d3:~<d2:3

          
        """
        if(isinstance(condition, context.Context)):
            condition = context._apply(condition, self)
        return repops_multi.Filter(self, condition, dim, mode) 

    def _getResultSlices(self, args={}, endpoint=True, log=False, debug=False):
        query = query_context.QueryContext(self, args, endpoint)
        if debug:
            return tuple(engines.debug_engine.run(query, log))
        else:
            return tuple(engines.select_engine.run(query, log))

    def __call__(self, **args):
        res = self._getResultSlices(**args)

        if(len(res) == 1):
            return res[0].data
        else:
            return tuple([slice.data for slice in res])

    def Detect(self, *args, **kwargs):
        """Detects types of slices, and casts result to this type
            
            :param only_unknown: Only detect slices with unknown types [default: False]

            :param allow_convert: Converts also types (e.g. bytes to integers/floats where possible) [default: True]

        """
        return repops.Detect(self, *args, **kwargs).Copy()

    # Overloaded functions. The r* functions are implemented because you
    # want to the same behaviour no matter no which side the known object
    # resides. (a + b == b + a)
    # add ( + )
    def __add__(self, other):
        if(isinstance(other, context.Context)):
            return other.__radd__(self)
        return repops_funcs.Add(self, other)
    
    def __radd__(self, other):
        return repops_funcs.Add(other, self)
    
    def __sub__(self, other):
        if(isinstance(other, context.Context)):
            return other.__radd__(self)
        return repops_funcs.Subtract(self, other)
    
    def __rsub__(self, other):
        return repops_funcs.Subtract(other, self)

    # multiplication ( * )
    def __mul__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rmul__(self)
        return repops_funcs.Multiply(self, other)
    
    def __rmul__(self, other):
        return repops_funcs.Multiply(other, self)
    
    # modulo ( % )
    def __mod__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rmod__(self)
        elif(isinstance(other, basestring)):
            return repops_dim.DimRename(self, other)
        elif(isinstance(other, tuple)):
            return repops_dim.DimRename(self, *other)
        elif(isinstance(other, dict)):
            return repops_dim.DimRename(self, **other)
        return repops_funcs.Modulo(self, other)
    
    def __rmod__(self, other):
        return repops_funcs.Modulo(other, self)

    # division ( / )
    def __div__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rdiv__(self)
        elif(isinstance(other, basestring)):
            return repops_slice.SliceRename(self, other)
        elif(isinstance(other, tuple)):
            return repops_slice.SliceRename(self, *other)
        elif(isinstance(other, dict)):
            return repops_slice.SliceRename(self, **other)
        return repops_funcs.Divide(self, other)
    
    def __rdiv__(self, other):
        return repops_funcs.Divide(other, self)
    
    def __floordiv__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rfloordiv__(self)
        elif(isinstance(other, basestring)):
            return repops_slice.Bookmark(self, other)
        elif(isinstance(other, tuple)):
            return repops_slice.Bookmark(self, *other)
        elif(isinstance(other, dict)):
            return repops_slice.Bookmark(self, **other)
        return repops_funcs.FloorDivide(self, other)
    
    def __rfloordiv__(self,other):
        return repops_funcs.FloorDivide(other, self)
        

    def __pow__(self,other):
        if(isinstance(other, context.Context)):
            return other.__rpow__(self)
        return repops_funcs.Power(self, other)
    
    def __rpow__(self, other):
        return repops_funcs.Power(other,self)

    # and operator ( & )
    def __and__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rand__(self)
        return repops_funcs.And(self, other)
    
    def __rand__(self, other):
        return repops_funcs.And(other, self)
    
    # or operator ( | )
    def __or__(self, other):
        if(isinstance(other, (context.Context, infix.Infix))):
            return other.__ror__(self)
        return repops_funcs.Or(self, other)

    def __ror__(self, other):
        return repops_funcs.Or(other, self)
    
    # exclusive-or operator ( ^ )
    def __xor__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rxor__(self)
        return repops_funcs.Xor(self, other)

    def __rxor__(self, other):
        return repops_funcs.Xor(other, self)

    # less-than ( < )
    def __lt__(self, other):
        if(isinstance(other, context.Context)):
            return other.__gt__(self)
        return repops_funcs.Less(self, other)

    # less-than-or-equals ( <= )
    def __le__(self, other):
        if(isinstance(other, context.Context)):
            return other.__ge__(self)
        return repops_funcs.LessEqual(self, other)

    # equals ( == )
    def __eq__(self, other):
        if(isinstance(other, context.Context)):
            return other.__eq__(self)
        return repops_funcs.Equal(self, other)

    # not-equals ( != )
    def __ne__(self, other):
        if(isinstance(other, context.Context)):
            return other.__ne__(self)
        return repops_funcs.NotEqual(self, other)

    # greater-than ( > )
    def __gt__(self, other):
        if(isinstance(other, context.Context)):
            return other.__lt__(self)
        return repops_funcs.Greater(self, other)

    # greater-than-or-equals ( >= )
    def __ge__(self, other):
        if(isinstance(other, context.Context)):
            return other.__le__(self)
        return repops_funcs.GreaterEqual(self, other)

    # plus prefix, used for table operators (i.e. ++, &+, etc.)
    def __pos__(self):
        return repops.PlusPrefix(self)
    
    def __invert__(self):
        return repops_funcs.Invert(self)
       
    def __abs__(self):
        return repops_funcs.Abs(self)
    
    def __neg__(self):
        return repops_funcs.Negative(self)

    def __nonzero__(self):
        self._checkState()

        if(len(self._slices) != 1):
            raise RuntimeError, "Cannot determine True/False status of a multi-slice representor"

        slice = self._slices[0]
        if(len(slice.dims) > 1):
            nself = repops_dim.FlatAll(self)
        else:
            nself = self

        if(len(slice.dims) > 0):
            nself = repops_funcs.All(nself)
        
        res = nself()
        return bool(res)

    def Cast(self, *newtypes, **kwds):
        """Cast data to new type. 

        Allowed formats:

        * single type for all slices

          >>> x.Cast("int32")

        * type for each slice

          >>> x.Cast(("int64","int8"))

        * type for named slices

          >>> x.Cast(your_slice="int8")

        """
        return repops_slice.SliceCast(self, *newtypes, **kwds)

    def Transpose(self,permute_idxs=(1,0)):
        """Transposes the dimensions of slices.
           Can only be applied to the common dimensions of slices.

           :param permute_idxs: index order of new dims.
                By default, performs matrix transpose, of first two dims, 
                i.e. permute_idxs=(1,0)

        """
        return repops_dim.PermuteDims(self,permute_idxs)

    def Flat(self, dim=-1,name=None):
        """Flattens (merges) a dimension with previous(parent) dim. 

        :param dim: Dim to flatten. By default, last dim.
        :param name: Name of merged dim. By default, merged name of two dimensions.
        """
        return repops_dim.Flat(self, name=name,dim=dim)

    def FlatAll(self, name=None):
        """Flattens all dimensions into one dimension.

        :param name: name of new merged dimension. By default, merged names of all dimensions.

        Note that this operation is slightly different from flat, in that it converts all slices
        to have 1 dimension, even those which have 0 dimensions. 

        Example::

            >>> x = Rep(([[1,2,3],[4,5,6]],'a'))
            
            Slices: | f0        | f1
            ------------------------------
            Type:   | int64     | bytes[1]
            Dims:   | d1:2<d2:3 |
            Data:   |           |
                    | [1 2 3]   | a
                    | [4 5 6]   |

            Dim order: d1:2<d2:3
            
            >>> x.Flat()
                     
            Slices: | f0      | f1
            ----------------------------
            Type:   | int64   | bytes[1]
            Dims:   | d1_d2:6 |
            Data:   |         |
                    | 1       | a
                    | 2       |
                    | 3       |
                    | 4       |
                    | 5       |
                    | 6       |

            Dim order: d1_d2:6

            >>> x.FlatAll()

            Slices: | f0      | f1      
            ----------------------------
            Type:   | int64   | bytes[1]
            Dims:   | d1_d2:6 | d1_d2:6 
            Data:   |         |         
                    | 1       | a       
                    | 2       | a       
                    | 3       | a       
                    | 4       | a       
                    | 5       | a       
                    | 6       | a       

            Dim order: d1_d2:6
   

        """
        return repops_dim.FlatAll(self,name=name)

    def SplitDim(self,lshape,rshape,lname=None,rname=None,dimsel=None):
        """Splits dim into two dimensions.

        :param lshape: Left shape (integer or array of lengths)
        :param rshape: Right dim shape
        :param lname: New name of left dimension (default:autogenerated).
        :param rname: New name of right dimension (default:autogenerated).
        :param dimsel: Dim to split (default: last common dimension).

        Example::
        
            >>> x = Rep([1,2,3,4,5,6])
            
            Slices: | data
            ---------------
            Type:   | int64
            Dims:   | d1:6
            Data:   |
                    | 1
                    | 2
                    | 3
                    | 4
                    | 5
                    | 6

            Dim order: d1:6
            
            >>> x.SplitDim(3,2)
            
            Slices: | data
            -----------------------
            Type:   | int64
            Dims:   | d2:3<d3:2
            Data:   |
                    | [1 2]
                    | [3 4]
                    | [5 6]

            Dim order: d2:3<d3:2
        
        """
        return repops_dim.SplitDim(self,lshape,rshape,lname,rname,dimsel)

    def Harray(self, name=None):
        """Combines slices into array.

           Example::
           
                >>> x = Rep([(1,2),(3,4),(5,6)])
                
                Slices: | f0    | f1
                -----------------------
                Type:   | int64 | int64
                Dims:   | d1:3  | d1:3
                Data:   |       |
                        | 1     | 2
                        | 3     | 4
                        | 5     | 6

                Dim order: d1:3
       
                >>> x.HArray()
                
                Slices: | f0_f1
                -------------------
                Type:   | int64
                Dims:   | d1:3<d2:2
                Data:   |
                        | [1 2]
                        | [3 4]
                        | [5 6]

                Dim order: d1:3<d2:2 

            Which is equivalent to this::

                >>> x.Get(HArray(_.f0, _.f1)) 

        """
        return repops_slice.HArray(self, name=name)

    def Shape(self):
        """Returns shape of all dimensions as slices in a representor object.
            
           Example::
    
                >>> x = Rep([[1,2,3],[4,5,6]])
                
                Slices: | data     
                -------------------
                Type:   | int64    
                Dims:   | d1:2<d2:3
                Data:   |          
                        | [1 2 3]  
                        | [4 5 6]  

                Dim order: d1:2<d2:3
                
                >>> x.Shape()

                Slices: | d1    | d2   
                -----------------------
                Type:   | int64 | int64
                Dims:   |       | d1:2 
                Data:   |       |      
                        | 2     | 3    
                        |       | 3    

                Dim order: d1:2

        """
        return repops_dim.Shape(self)

    def GroupBy(self, *args, **kwargs):
        """Groups data on the content of one or more slices.
        
        :param flat: Allows one to indicate which slices should not be grouped. 
        
        Example::

            >>> x = Rep(([1,1,2,2,3,3,4,4],[1,2,1,2,1,2,1,2],[1,2,3,4,1,2,3,4]))

            Slices: | f0    | f1    | f2
            -------------------------------
            Type:   | int64 | int64 | int64
            Dims:   | d1:8  | d1:8  | d1:8
            Data:   |       |       |
                    | 1     | 1     | 1
                    | 1     | 2     | 2
                    | 2     | 1     | 3
                    | 2     | 2     | 4
                    | 3     | 1     | 1
                    | 3     | 2     | 2
                    | 4     | 1     | 3
                    | 4     | 2     | 4

            Dim order: d1:8

            >>> x.GroupBy(_.f0)
            
            Slices: | f0    | f1          | f2
            -------------------------------------------
            Type:   | int64 | int64       | int64
            Dims:   | gf0:* | gf0:*<gd1:~ | gf0:*<gd1:~
            Data:   |       |             |
                    | 1     | [1 2]       | [1 2]
                    | 2     | [1 2]       | [3 4]
                    | 3     | [1 2]       | [1 2]
                    | 4     | [1 2]       | [3 4]

            Dim order: gf0:*<gd1:~            

        Note how slice f0 has now only unique values, and how slices f1 and f2 have now two dimensions, grouped per unique value in f0. 
        One can also group on multiple slices at once::
            
            >>> x.GroupBy((_.f1, _.f2))

            Slices: | f0            | f1            | f2           
            -------------------------------------------------------
            Type:   | int64         | int64         | int64        
            Dims:   | gdata:*<gd1:~ | gdata:*<gd1:~ | gdata:*<gd1:~
            Data:   |               |               |              
                    | [2 4]         | [2 2]         | [4 4]        
                    | [2 4]         | [1 1]         | [3 3]        
                    | [1 3]         | [1 1]         | [1 1]        
                    | [1 3]         | [2 2]         | [2 2]        

            Dim order: gdata:*<gd1:~

        This groups the data such that the combination of f1 and f2 is unique. 
        This actually equivalent to::
        
            >>> x.GroupBy(_.Get(_.f1, _.f2).Tuple())
        
        That is, '(_.f1, _.f2)' signifies that one wants to get the tuple of f1 and f2, which looks like this::

            >>> x.Get((_.f1, _.f2))

            Slices: | data                
            ------------------------------
            Type:   | (f1=int64, f2=int64)
            Dims:   | d1:8                
            Data:   |                     
                    | (1, 1)              
                    | (2, 2)              
                    | (1, 3)              
                    | (2, 4)              
                    | (1, 1)              
                    | (2, 2)              
                    | (1, 3)              
                    | (2, 4)              

            Dim order: d1:8
        
        Instead of grouping on combinations of slices, one can also group on multiple slices individually::
        
            >>> x.GroupBy(_.f0, _.f1)
        
            Slices: | f0    | f1    | f2               
            -------------------------------------------
            Type:   | int64 | int64 | int64            
            Dims:   | gf0:* | gf1:* | gf0:*<gf1:*<gd1:~
            Data:   |       |       |                  
                    | 1     | 1     | [[1] [2]]        
                    | 2     | 2     | [[3] [4]]        
                    | 3     |       | [[1] [2]]        
                    | 4     |       | [[3] [4]]        

            Dim order: gf0:*<gf1:*<gd1:~


        Note that f0 and f1 now have two separate dimensions, while f2 has both these dimensions, and an extra 'group' dimension (like before). In this 
        case, the gd1 dimensions is always of length 1, as there are only unique values in f2 for every pair of f0, f1. 

        
        Of course, one remove such an extra dim using filtering, e.g.::

            >>> x.GroupBy(_.f0, _.f1)[...,0]

            Slices: | f0    | f1    | f2
            -------------------------------------
            Type:   | int64 | int64 | int64
            Dims:   | gf0:* | gf1:* | gf0:*<gf1:*
            Data:   |       |       |
                    | 1     | 1     | [1 2]
                    | 2     | 2     | [3 4]
                    | 3     |       | [1 2]
                    | 4     |       | [3 4]

            Dim order: gf0:*<gf1:*

        However, one can also already indicate to groupby that some slices do not have to be grouped, using the 'flat' parameter. Note for example,
        how for this case, values in f1 and f2 are for every group the same::

            >>> x.GroupBy((_.f1, _.f2))

        We can prevent te grouping of f1 and f2 using flat::

            >>> x.GroupBy((_.f1, _.f2),flat=['f1','f2'])

            Slices: | f0            | f1      | f2     
            -------------------------------------------
            Type:   | int64         | int64   | int64  
            Dims:   | gdata:*<gd1:~ | gdata:* | gdata:*
            Data:   |               |         |        
                    | [2 4]         | 2       | 4      
                    | [2 4]         | 1       | 3      
                    | [1 3]         | 1       | 1      
                    | [1 3]         | 2       | 2      

       
        Or in case of the multi-dimensional group::

            >>> x.GroupBy(_.f0, _.f1, flat='f2') 

            Slices: | f0    | f1    | f2         
            -------------------------------------
            Type:   | int64 | int64 | int64      
            Dims:   | gf0:* | gf1:* | gf0:*<gf1:*
            Data:   |       |       |            
                    | 1     | 1     | [1 2]      
                    | 2     | 2     | [3 4]      
                    | 3     |       | [1 2]      
                    | 4     |       | [3 4]      

            Dim order: gf0:*<gf1:*
                            
        Note that f2 is now along every dimension non-unique.

        However, one might also have a case in which a slice is non-unique for just a single slice in a multi-dimensional group, e.g.::


            >>> x.Get(_.f0, _.f1, _.f2, _.f1/'f3').GroupBy(_.f0, _.f1, flat=['f2','f3'])        

        Here, we copied slice f1, calling it 'f3'. Next, we specified that it should be flat. But note that this slice is still unique along dimension gf0...
        Here, we can use a more advanced format for the flat parameter, in which one can specify w.r.t. to which slices a slice should be grouped::

            >>> x.Get(_.f0, _.f1, _.f2, _.f1/'f3').GroupBy(_.f0, _.f1, flat={('f0','f1'):'f2','f1':'f3'})

            Slices: | f0    | f1    | f2          | f3
            ---------------------------------------------
            Type:   | int64 | int64 | int64       | int64
            Dims:   | gf0:* | gf1:* | gf0:*<gf1:* | gf1:*
            Data:   |       |       |             |
                    | 1     | 1     | [1 2]       | 1
                    | 2     | 2     | [3 4]       | 2
                    | 3     |       | [1 2]       |
                    | 4     |       | [3 4]       |


        This specifies that f2 should be flat, while keeping the group for both grouping slices, and 'f3' should be flat, while keeping the group only for the second group slice. 
        It is equivalent to::

            >>> x.Get(_.f0, _.f1, _.f2, _.f1/'f3').GroupBy(_.f0, _.f1, flat={(0,1):'f2',1:'f3'})        

        """

        flat = kwargs.pop("flat", {})
        group_source = self.Get(*args, **kwargs)

        default_dim = tuple(range(len(args)))
        
        if(isinstance(flat,dict)):
            pass
        elif(isinstance(flat, (list,tuple))):
            flat = {default_dim:flat}
        else:
            flat = {default_dim:[flat]}
            
        return repops_multi.Group(self, group_source, flat)
   
    def Join(self, other, cond):
        """Join allows you to take the cartesian product of two dimensions, and 
        filter them on some condition.
        
        Example::
        
            >>> x = Rep([1,2,3,4,5])
            >>> y = Rep([3,4,5,6,7,8])

            >>> x.Join(y, x >= y)
       
            Slices: | data       | data      
            ---------------------------------
            Type:   | int64      | int64     
            Dims:   | d1a_fd1b:* | d1a_fd1b:*
            Data:   |            |           
                    | 3          | 3         
                    | 4          | 3         
                    | 4          | 4         
                    | 5          | 3         
                    | 5          | 4         
                    | 5          | 5         

            Dim order: d1a_fd1b:*

        The join operation generates all possible pairs of values out of x and y, and then filters them on x >= y. 
        Note that you can also use the following equivalent forms::
            
            >>> Join(x, y, x >= y)

            >>> x |Join(x >= y)| y
        
        One can also make this condition also more complex, eg::

            >>> x = Rep([[1,2,3],[4,5,6]])

            Slices: | data     
            -------------------
            Type:   | int64    
            Dims:   | d1:2<d2:3
            Data:   |          
                    | [1 2 3]  
                    | [4 5 6]  

            Dim order: d1:2<d2:3

            >>> x.Join(y, x <= y)

            Slices: | data                                  | data                                 
            ---------------------------------------------------------------------------------------
            Type:   | int64                                 | int64                                
            Dims:   | d1:2<d2_fd1a:~                        | d1b:2<d2_fd1a:~                      
            Data:   |                                       |                                      
                    | [1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3] | [3 4 5 6 7 8 3 4 5 6 7 8 3 4 5 6 7 8]
                    | [4 4 4 4 4 5 5 5 5 6 6 6]             | [4 5 6 7 8 5 6 7 8 6 7 8]            

            Dim order: d1:2<d2_fd1a:~

            >>> x.Join(y, x.Sum() <= y * y)
            
            Slices: | data            | data
            --------------------------------------
            Type:   | int64           | int64
            Dims:   | d1b_fd1a:*<d2:3 | d1b_fd1a:*
            Data:   |                 |
                    | [1 2 3]         | 3
                    | [1 2 3]         | 4
                    | [1 2 3]         | 5
                    | [1 2 3]         | 6
                    | [1 2 3]         | 7
                    | [1 2 3]         | 8
                    | [4 5 6]         | 4
                    | [4 5 6]         | 5
                    | [4 5 6]         | 6
                    | [4 5 6]         | 7
                    | [4 5 6]         | 8

            >>> x.Join(y, y |In| x)

            Slices: | data            | data      
            --------------------------------------
            Type:   | int64           | int64     
            Dims:   | d1b_fd1a:*<d2:3 | d1b_fd1a:*
            Data:   |                 |           
                    | [1 2 3]         | 3         
                    | [4 5 6]         | 4         
                    | [4 5 6]         | 5         
                    | [4 5 6]         | 6         

            Dim order: d1b_fd1a:*<d2:3

        The use of context operators is a bit more complex with Join operations, as the context can refer to both sources. 
        The context operator therefore refers to the combination of both. If there is a conflict in slice names (like in the previous examples), one can 
        refer to both slices using the 'L' and 'R' bookmark (see Combine operation)::
        
            >>> x.Join(y, _.L == _.R)

            Slices: | data           | data           
            ------------------------------------------
            Type:   | int64          | int64          
            Dims:   | d1:2<d2_fd1a:~ | d1b:2<d2_fd1a:~
            Data:   |                |                
                    | [3]            | [3]            
                    | [4 5 6]        | [4 5 6]        

            Dim order: d1:2<d2_fd1a:~
          
        .. warning:
            
            The current implementation of Join is not very efficient, and thus can cause problems if the dataset is very large. For equi-joins 
            (where the condition is an equality condition between slices in both sources), use the "Match" operation.
     
        """
        
        return repops_multi.Join(self, other, cond)
    
    def Match(self, other, lslice=None, rslice=None, jointype="inner", merge_same=False, mode="dim"):
        """Match allows you to take the cartesian product of two dimensions, and 
        filter them on an equality condtion.

        :param other:  ibidas representor to match with
        :param lslice: slice in self to perform equality condition on (see .Get for allowed parameter values). Default: use a slice which has same name in both sources (there should be only 1 slice pair with this property). 
        :param rslice: slice in other to perform equality condition on (see .Get for allowed parameter values). Default: use same name as lslice. 
        :param jointype: choose between 'inner', 'left','right', or 'full' equijoin. Default: inner
        :param merge_same: False, 'equi' or 'all'. Default: False
        :param mode: Type of broadcasting, 'dim' or 'pos', i.e. respectively on identity or position. Default: 'dim'. 

        Examples::
            
            >>> x = Rep([('a',1), ('b', 2), ('c',3), ('c', 4)])

            Slices: | f0       | f1
            --------------------------
            Type:   | bytes[1] | int64
            Dims:   | d1:4     | d1:4
            Data:   |          |
                    | a        | 1
                    | b        | 2
                    | c        | 3
                    | c        | 4

            Dim order: d1:4

            >>>  y = Rep([('a','test1'),('d','test2'), ('c', 'test3')])
            
            Slices: | f0       | f1
            -----------------------------
            Type:   | bytes[1] | bytes[5]
            Dims:   | d1:3     | d1:3
            Data:   |          |
                    | a        | test1
                    | d        | test2
                    | c        | test3

            Dim order: d1:3

           
            >>> x.Match(y, _.f0, _.f0) 

            Slices: | f0       | f1    | f1      
            -------------------------------------
            Type:   | bytes[1] | int64 | bytes[5]
            Dims:   | d1:*     | d1:*  | d1:*    
            Data:   |          |       |         
                    | a        | 1     | test1   
                    | c        | 3     | test3   
                    | c        | 4     | test3   

            Dim order: d1:*

        The f0 slices of the 'x' and 'y' have been collapsed into a single slice, as they had the same name and content (as imposed bui
        the equality condition). 
        
        Note that this call is equivalent to::
            
            >>> x |Match(_.f0, _.f0)| y

        Or, because both slices are named similarly::
            
            >>> x |Match(_.f0)| y

        To access similarly named slices from the left or right operand, use the bookmarks as defined by the Combine operation (see documentation there)::
        
            >>> (x |Match(_.f0)| y).R.f1

            Slices: | f1      
            ------------------
            Type:   | bytes[5]
            Dims:   | d1:*    
            Data:   |         
                    | test1   
                    | test3   
                    | test3   

            Dim order: d1:*  

        
        The join type by default is 'inner', which means that only rows which are similar in both slices are kept. One can also
        use the 'left', 'right' or 'full' join types. In these cases, unmatched rows in respectively the left, right and both source(s) are also kept::

            >>> x |Match(_.f0, join_type=='left')| y          
    
            Slices: | f0       | f1    | f0        | f1       
            --------------------------------------------------
            Type:   | bytes[1] | int64 | bytes?[1] | bytes?[5]
            Dims:   | d1:*     | d1:*  | d1:*      | d1:*     
            Data:   |          |       |           |          
                    | a        | 1     | a         | test1    
                    | c        | 3     | c         | test3    
                    | c        | 4     | c         | test3    
                    | b        | 2     | --        | --       

            Dim order: d1:*

            >>> x |Match(_.f0, join_type=='full')| y         
     
            Slices: | f0        | f1     | f0        | f1       
            ----------------------------------------------------
            Type:   | bytes?[1] | int64? | bytes?[1] | bytes?[5]
            Dims:   | d1:*      | d1:*   | d1:*      | d1:*     
            Data:   |           |        |           |          
                    | a         | 1      | a         | test1    
                    | c         | 3      | c         | test3    
                    | c         | 4      | c         | test3    
                    | b         | 2      | --        | --       
                    | --        | --     | d         | test2    

            Dim order: d1:*

        Sometimes in these cases, it is usefull to merge the slices that have similar information, in this case both 'f0' slices. This
        can be accomplished using the 'merge_same' parameter::

            >>> x |Match(_.f0, join_type=='full', merge_same='equi')| y

            Slices: | f0        | f1     | f1
            ----------------------------------------
            Type:   | bytes?[1] | int64? | bytes?[5]
            Dims:   | d1:*      | d1:*   | d1:*
            Data:   |           |        |
                    | a         | 1      | test1
                    | c         | 3      | test3
                    | c         | 4      | test3
                    | b         | 2      | --
                    | d         | --     | test2

            Dim order: d1:*
        
        The value 'equi' selects the slices used for the equality condition. An alternative is to call with 
        a tuple of the slice names that should be merged::
            
            >>> x |Match(_.f0, jointype='full', merge_same =('f0',))| y 
        
        Another case is when one wants to merge slices with dissimilar names. This can be accomplished by using a nested tuple::
            
            >>> x |Match(_.f0, jointype='full', merge_same =(('f0','f0'),))| y 
                    
        Finally, one can also merge on all slices with the same names, by setting merge_same to 'all' or True. For the current example,
        this would generate an error, because slices 'f1' and 'f1' have conflicting content for the same rows::

            >>> x |Match(_.f0, jointype='full', merge_same =True)| y
            RuntimeError: Found unequal values during merge: 1 != test1
        
        The final parameter, 'mode', can only be illustrated with a slightly more complicated example, in which we have multiple dimensions::

            >>> x = Rep([[1,2],[1,2,3]])    
            
            Slices: | data     
            -------------------
            Type:   | int64    
            Dims:   | d1:2<d2:~
            Data:   |          
                    | [1 2]    
                    | [1 2 3]  

            Dim order: d1:2<d2:~ 


            >>> y = Rep([[2,3,4],[1,3,4]])

            Slices: | data
            -------------------
            Type:   | int64
            Dims:   | d1:2<d2:3
            Data:   |
                    | [2 3 4]
                    | [1 3 4]

            Dim order: d1:2<d2:3

        Matching these datasets to each other, will match them on dimensions 'd2' in both datasets (which get renamed to d2a and d2b)::

            >>> x |Match| y
            Slices: | data
            -------------------------------
            Type:   | int64
            Dims:   | d1a:2<d1b:2<d2a_d2b:~
            Data:   |
                    | [[2] [1]]
                    | [[2 3] [1 3]]

            Dim order: d1a:2<d1b:2<d2a_d2b:~
        
        Note that the dataset has three dimensions, a two by two matrix of the dimensions 'd1' in both datasets, with nested lists of the Match results of each pair of rows
        of both datasets. 

        But, maybe we intended for dimensions 'd1' in both datasets to be matched to each other, although they have not the same identity. With 'positional' broadcasting,
        we match dimensions on position, which for this case is both the same (1 dimension before the matching dimension). 

            >>> x |Match(mode='pos')| y

            slices: | data
            -------------------------
            Type:   | int64
            Dims:   | d1a:2<d2a_d2b:~
            Data:   |
                    | [2]
                    | [1 3]

            Dim order: d1a:2<d2a_d2b:~

        Note that both d1 dimensions have now be matched to each other, and a Match is done between only [1,2] and [2,3,4], and [1,2,3] and [1,3,4], instead of all possible pairs of rows. 
        """

        return repops_multi.Match(self, other, lslice, rslice, jointype, merge_same, mode)
    
    def Intersect(self, other, slices=COMMON_POS, dims=LASTCOMMONDIM, mode='dim'):
        """Intersect compares dataset A and B, given only rows from A that occur also in B. 
            
           :param other: Other dataset to compare with
           :param slices: Specify on which slices an intersection should be performed. COMMON_POS (pair slices with common position), COMMON_NAME (pair slices with common names) or a tuple with for each
                source a (tuple of) slice name(s).  Default: COMMON_POS. 
           :param dims: Specify across which dimensions an intersection should be performed. Default: last common dim (-1L). . 
           :param mode: 'dim' or 'pos'. Type of broadcasting (dimension identity or positional). Default: 'dim'


        """        
        
        return repops_multi.Intersect(self, other, slices=slices, dims=dims, mode=mode)
    
    def Except(self, other, slices=COMMON_POS, dims=LASTCOMMONDIM, mode='dim'):
        """Except compares dataset A and B, given only rows from A that occur not in B. For further documentation,
           see 'Intersect'"""
        return repops_multi.Intersect(self, other, slices=slices, dims=dims, mode=mode)
    
    def Difference(self, other, slices=COMMON_POS, dims=LASTCOMMONDIM, mode='dim'):
        """Difference compares dataset A and B, given only rows that occur not in both. For further documentation, 
            see 'Intersect'"""
        return repops_multi.Intersect(self, other, slices=slices, dims=dims, mode=mode)
   
    def Union(self, other, slices=COMMON_POS, dims=LASTCOMMONDIM, mode='dim'):
        """Union compares dataset A and B, given all unique rows that occur in either or both datasets. For further
            documentation, see 'Intersect'"""
        return repops_multi.Intersect(self, other, slices=slices, dims=dims, mode=mode)

    def Replace(self, slice, translator, fromslice=0, toslice=1):
        return repops_multi.Replace(self, slice, translator, fromslice, toslice) 

    def ReplaceMissing(self, def_value=NOVAL):
        return repops_funcs.ReplaceMissing(self, def_value=def_value)

    def Merge(self, other):
        return repops_funcs.Merge(self, other)

    def Rename(self, *names, **kwds):
        """Rename slices.

        :param names: names without keywords. Number of names should match number of slices.
        :param kwds: names with keywords, e.g.  f0="genes". Number does *not* have to match number of slices.

        Examples::
            
            >>> na = a.Rename("genes","scores")
            >>> na = a.Rename(f0 = "genes", f1 = "scores")
        
        Shortcut:
            One can use the division operation to rename slices.

            * tuple:
                Number of names in tuple should match number of slices.

                >>> a/("genes", "scores")

            * dict:
                Similar to keywords.

                >>> a/{"f0": "genes"}

            * str:
                Can only be used if representor object consists of single slice.

                >>> a.f0/"genes"
        """ 
        return repops_slice.SliceRename(self, *names, **kwds)
    
    def DimRename(self, *names, **kwds):
        """Rename dimensions. Similar to ``Rename`` for slices.

        Shortcut: use % operator
        """
        return repops_dim.DimRename(self, *names, **kwds)
  
    def Bookmark(self, *names, **kwds):
        """Bookmarks slices with a name. 
        Slices can later be accessed using attribute access, 
        with axis indicator "B". 

        Example:
            >>> x = x.Bookmark("myslices")
            >>> x.myslices  
            >>> x.Bmyslices   #in case there is also a slice named 'myslices'
        """
        return repops_slice.Bookmark(self, *names, **kwds)

    def Each(self, eachfunc, dtype=rtypes.unknown):
        return repops_funcs.Each(self, eachfunc=eachfunc, dtype=dtype)
    
    def Pos(self, dim=None):
        return repops_funcs.Pos(self, dim)
    
    def CumSum(self, dim=None):
        return repops_funcs.CumSum(self, dim)
   
    def Argsort(self, dim=None, descend=False):
        return repops_funcs.Argsort(self, dim, descend=descend)
    
    def Rank(self, dim=None, descend=False):
        return repops_funcs.Rank(self, dim, descend=descend)
   
    def Argunique(self, dim=None):
        return repops_funcs.Argunique(self, dim)

    def Sum(self, dim=None):
        return repops_funcs.Sum(self, dim)

    def Max(self, dim=None):
        return repops_funcs.Max(self, dim)

    def Min(self, dim=None):
        return repops_funcs.Min(self, dim)
    
    def Argmax(self, dim=None):
        return repops_funcs.Argmax(self, dim)

    def Argmin(self, dim=None):
        return repops_funcs.Argmin(self, dim)

    def Mean(self, dim=None):
        return repops_funcs.Mean(self, dim)
    
    def Median(self, dim=None):
        return repops_funcs.Median(self, dim)
    
    def Std(self, dim=None):
        return repops_funcs.Std(self, dim)
     
    def Any(self,dim=None):
        return repops_funcs.Any(self,dim)

    def All(self,dim=None):
        return repops_funcs.All(self,dim)

    def Count(self):
        return repops_funcs.Count(self)

    def __len__(self):
        self._checkState()
        assert len(set([s.dims[0] for s in self._slices])) == 1, "To execute len, there should be only one root dimension"
        return self.Get(0).Array(1).Count()()
        
    def Set(self, dim=None):
        return repops_funcs.Set(self,dim)
    
    def Unique(self, dim=None):
        return repops_funcs.Unique(self,dim)

    def In(self, arrays):
        return repops_funcs.Within(self,arrays)

    def Contains(self, elems):
        return repops_funcs.Contains(self,elems)

    def Array(self, tolevel=None):
        """Packages dimension into array type"""
        return repops_dim.Array(self, tolevel = tolevel)

    def Level(self, tolevel):
        """Bring all slices to same dimension height throug packing and broadcasting"""
        return repops_dim.Level(self, tolevel)

    def Tuple(self):
        """Combines slices into a tuple type"""
        return repops_slice.Tuple(self)
    
    def Dict(self, with_missing=False):
        """Combines slices into a tuple type"""
        return repops_slice.Dict(self, with_missing=with_missing)
    
    def IndexDict(self):
        """Combines slices into a tuple type"""
        return repops_slice.IndexDict(self)

    def TakeFrom(self, other, allow_missing=False,keep_missing=False):
        return repops_multi.Take(other, self, allow_missing,keep_missing)

    def ToPython(self, **args):
        """Converts data into python data structure"""
        return repops_slice.ToPythonRep(self)(**args)

    def Sort(self, *slices, **kwargs):
        """Performs sort on data.
        
        Example:

        *  Sort slices in x on all slices. If multiple slices, combines into tuple, then sort it.

           >>> x.Sort()

        *  Sort x on slice f1

           >>> x.Sort(_.f1)

        * Sort x on slice f1, f3. 

          >>> x.Sort(_.f1, _.f3) 
        
        For other possible sort slice selection formats, see ``get`` function. 

        """
        descend = kwargs.pop("descend",False)

        if(slices or kwargs):
            sortsource = self.Get(*slices,**kwargs)
            return repops_multi.Sort(self, sortsource, descend=descend)
        else:
            return repops_multi.Sort(self, descend=descend)

    def Unique(self, *slices, **kwargs):
        if(slices or kwargs):
            sortsource = self.Get(*slices,**kwargs)
            return repops_multi.Unique(self, sortsource)
        else:
            return repops_multi.Unique(self)

    def To(self, *slices, **kwargs):
        return repops_slice.To(self, *slices, **kwargs)
              
              
    def AddSlice(self, name, data, dtype=None):
        return repops_slice.AddSlice(self, data, name, dtype)


    def Get(self, *slices, **kwds):
        """Select slices in a new representor, combine with other slices.

        :param slices: Can be various formats:

             * str:  selects slice with this name.
                     Special symbols:

                     - "*":  all slices

                     - "~": all slices with names not yet in previous ``get`` parameters are selected.

                     - "#": select first slice if all slices have a common dimension. 

             * int:   selects slice with this index

             * slice: selects slices with these indexes (note: a *Python* slice object, not an Ibidas slice)

             * representor: adds this representor slices

             * context: apply to self, adds resulting slices

             * tuple: applies get to elements in tuple, creates tuple slice from resulting slices (see .tuple() function)
             
             * list:  selected slices within list are packed using array function

        :param kwds: Same as slices, but also assigns slice names. 
        
        Examples:
            
            * str

              >>> a.Get("f0","f3")

            * int
    
              >>> a.Get(0, 3)

            * slice

              >>> a.Get(slice(0,3))
            
            * representor

              >>> a.Get(a.f0 + 3, a.f3 == "gene3",  Rep(3))

            * context

              >>> a.perform_some_operation().Get(_.f0 + 3, _.f3 == "gene3")
            
            * tuple

              >>> a.Get((_.f0, _.f1), _.f3)

            * list

              >>> a.Get([_.f0])

        """
        return repops_slice.Project(self,*slices,**kwds)

    def Without(self, *slices):
        return repops_slice.Unproject(self,*slices)

    def Elems(self, name=None):
        """Unpacks array type into dimension"""
        return repops_dim.UnpackArray(self, name)
    Elem = Elems

    def Fields(self, name=None):
        """Unpacks tuple type into slices"""
        return repops_slice.UnpackTuple(self, name)

    def GetRepeatedSliceNames(self):
        Pos = repops.delayable()(repops_funcs.Pos)
        Rep = python.Rep
        self._checkState()

        r = Rep(self.Names).Get(_/'name', Pos()/'pos').GroupBy(_.name)
        return r[_.pos.Count() > 1].Dict().ToPython()
        
        
            


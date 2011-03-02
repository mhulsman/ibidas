import copy
from constants import *
import representor
_delay_import_(globals(),"ops")
_delay_import_(globals(),"itypes","dimpaths")
_delay_import_(globals(),"utils","context")
_delay_import_(globals(),"repops_multi")

def delayable(default_params=["*"], nsources=1):
    """Function to enable the delay of single source operations.

    default_params: slice to apply the function to if no parameter is
                   given.

    Delayable functions should have as first param a source field. 
    If source is:
    - a Representor object: functions like normal function
    - a Context object: creates a context object with the function applied
    - any other: creates a context object with the parameter for source 
      given to the get function of the representor the context object
      should be applied to.
    - none: replaces with the argument in default_slice
    Other parameters of the function can still be given, either by 
    param order or keyword.
    Examples:
    data = Rep(data) (f1, f2, f3 slices)

    data[pos(data.f0) < 10] == data[pos(_.f0) < 10] == \
        data[pos("#") < 10] == data[pos() < 10]

    """
    def new(func):
        def delayable_function(*params, **kwds):
            if(not params):
                return _.Get(*default_params)._call(func, **kwds)
            
            if(nsources == UNDEFINED):
                xnsources = len(params)
            else:
                xnsources = nsources

            need_get = False

            for i in xrange(xnsources):
                if(not isinstance(params[i], representor.Representor)):
                    need_get = True
            if(need_get):
                return _.Get(*params[:xnsources])._call(func, *params[xnsources:], **kwds)
            else:
                if(xnsources > 1):
                    return func(repops_multi.Combine(*params[:xnsources]), *params[xnsources:], **kwds)
                else:
                    return func(*params, **kwds)
        return delayable_function
    return new

class UnaryOpRep(representor.Representor):
    def __init__(self, source, *args, **kwds):
        assert isinstance(source,representor.Representor), "Source should be a representor"
        self._source = source
        self._params = (args,kwds)
        self._process(source,*args, **kwds)

    def _process(self, source, *args, **kwds):
        if not source._slicesKnown():
            return
        return self._sprocess(source, *args, **kwds)

    def _sprocess(self, source, *args, **kwds):
        return self._initialize(source._slices)

class MultiOpRep(representor.Representor):
    def __init__(self, sources, *args,**kwds):
        assert isinstance(sources,tuple), "Sources should be a tuple"
        self._sources = sources
        self._params = (args,kwds)
        self._process(sources,*args, **kwds)
    
    def _process(self, sources, *args, **kwds):
        if not all([source._slicesKnown() for source in sources]):
            return
        return self._sprocess(sources, *args, **kwds)      
    
    def _sprocess(self, sources,*args, **kwds):
        raise RuntimeError, "Process function should be overloaded for " + str(type(self))
    

class Fixate(UnaryOpRep):#{{{
    """Operation used by optimizer to fixate end of tree,
    such that there are no exception situations, and slice retrieval
    is handled correctly."""

    def _sprocess(self, source):
        nslice = ops.FixateOp(source._slices)
        self._initialize((nslice,))
    #}}}

class Gather(Fixate):#{{{
    """Operation used by optimizer to fixate end of tree,
    such that there are no exception situations, and slice retrieval
    is handled correctly."""
    def _sprocess(self, source):
        nslice = ops.GatherOp(source._slices)
        self._initialize((nslice,))#}}}

class PlusPrefix(UnaryOpRep):#{{{
    pass
#}}}

class ApplyFuncRep(UnaryOpRep):
    """Applies slice class in `slicecls` to every field in source.

    :param dim_selector: select dimensions on which to apply the slicecls
    """

    def _sprocess(self,source,func,*params,**kwds):
        nslices = func(source._slices, *params, **kwds)
        return self._initialize(nslices)


def apply_slice(slices, slicecls, dim_selector, *params, **kwds):
    """Applies slice class in `slicecls` to every field in source.

    :param dim_selector: select dimensions on which to apply the slicecls
    """
    if(not dim_selector is None):
        dim_selector = dimpaths.identifyUniqueDimPath(slices, dim_selector)
        nslices = []                       #new active slices
        for slice in slices:
            if(dim_selector in slice.dims):
                nslice = slicecls(slice, *params, **kwds)
            else:
                nslice = slice
            nslices.append(nslice)
    else:
        nslices = [slicecls(slice,*params,**kwds) for slice in slices]
    return tuple(nslices)



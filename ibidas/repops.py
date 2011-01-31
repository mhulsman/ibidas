import copy
from constants import *
import representor
_delay_import_(globals(),"slices")
_delay_import_(globals(),"dim_helpers")
_delay_import_(globals(),"utils","context")


def delayable(default_slice="*"):
    """Function to enable the delay of single source operations.

    default_slice: slice to apply the function to if no parameter is
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
    data = rep(data) (f1, f2, f3 slices)

    data[pos(data.f0) < 10] == data[pos(_.f0) < 10] == \
        data[pos("#") < 10] == data[pos() < 10]

    """
    def new(func):
        def delayable_function(*params, **kwds):
            if(not params):
                if('source' in kwds):
                    params = (kwds['source'],)
                    del kwds['source']
                else:
                    return _.get(default_slice)._call(func, **kwds)
            if(not isinstance(params[0], representor.Representor)):
                if(isinstance(params[0], context.Context)):
                    return params[0]._call(func, *params[1:], **kwds)
                else:
                    return _.get(params[0])._call(func, *params[1:], **kwds)
            return func(*params, **kwds)
        return delayable_function
    return new

class UnaryOpRep(representor.Representor):
    def __init__(self, source, *args, **kwds):
        assert isinstance(source,representor.Representor), "Source should be a representor"
        self._source = source
        self._params = (args,kwds)
        self._process(source,*args, **kwds)

    def _process(self, source):
        if not source._state & RS_SLICES_KNOWN:
            return
        return self._initialize(source._slices, source._state)
    
class MultiOpRep(representor.Representor):
    def __init__(self, sources, *args,**kwds):
        assert isinstance(sources,tuple), "Sources should be a tuple"
        self._sources = sources
        self._params = (args,kwds)
        self._process(sources,*args, **kwds)
    
    def _process(self, sources):
        raise RuntimError, "Process function should be overloaded for " + str(type(self))
    

class Fixate(UnaryOpRep):#{{{
    """Operation used by optimizer to fixate end of tree,
    such that there are no exception situations, and slice retrieval
    is handled correctly."""
    pass

    #}}}

class PlusPrefix(UnaryOpRep):#{{{
    pass
#}}}

class ApplyFuncRep(UnaryOpRep):
    """Applies slice class in `slicecls` to every field in source.

    :param dim_selector: select dimensions on which to apply the slicecls
    """

    def _process(self,source,func,*params,**kwds):
        if not source._state & RS_ALL_KNOWN:
            return
        nslices = func(source._slices, *params, **kwds)
        return self._initialize(nslices)


def apply_slice(slices, slicecls, dim_selector, *params, **kwds):
    """Applies slice class in `slicecls` to every field in source.

    :param dim_selector: select dimensions on which to apply the slicecls
    """
    if(not dim_selector is None):
        dim_selector = dim_helpers.identifyUniqueDimPath(slices, dim_selector)
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

def frozen(slices):
    return apply_slice(slices, slices.ensure_frozen, None)

def converted(slices):
    return apply_slice(slices, slices.ensure_converted, None)


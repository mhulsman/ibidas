import copy

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

class OpRep(representor.Representor):
    def __init__(self, sources, all_slices, active_slices):
        representor.Representor.__init__(self, all_slices, 
                                               active_slices)
        self._sources = sources

    def _copyquery(self):
        res = copy.copy(self)
        res._sources = tuple([source._copyquery() for source in res._sources])
        return res

class UnaryOpRep(OpRep):
    pass

class MultiOpRep(OpRep):
    pass

class fixate(UnaryOpRep):#{{{
    """Operation used by optimizer to fixate end of tree,
    such that there are no exception situations, and slice retrieval
    is handled correctly."""

    def __init__(self, source):
        source = normal(source)
        UnaryOpRep.__init__(self, (source,), source._all_slices, 
                     source._active_slices)#}}}

class plusprefix(UnaryOpRep):#{{{
    def __init__(self, source):
        UnaryOpRep.__init__(self, (source,), source._all_slices, 
             source._active_slices)#}}}

def apply_slice(source, slicecls, dim_selector, *params, **kwds):
    """Applies slice class in `slicecls` to every field in source.

    Parameters
    ----------
    dim_selector: select dimensions on which to apply the slicecls
    params, kwds: extra params
    """
    if(not dim_selector is None):
        dim_selector = dim_helpers.identifyDimPath(source, dim_selector)
    
    nactive_slices = []                       #new active slices
    for slice in source._active_slices:
        if(dim_selector is None):
            nslice = slicecls(slice, *params, **kwds)
        elif(dim_helpers.sliceHasDimPath(slice, dim_selector)):
            nslice = slicecls(slice, *params, **kwds)
        else:
            nslice = slice
        nactive_slices.append(nslice)

    all_slices = source._all_slices.copy()
    for slice in nactive_slices:
        all_slices[slice.id] = slice

    #initialize object attributes
    self = UnaryOpRep((source,), all_slices, 
        tuple(nactive_slices))
    return self

def frozen(source):
    return apply_slice(source, slices.ensure_frozen, None)

def normal(source):
    return apply_slice(source, slices.ensure_normal, None)

def normal_or_frozen(source):
    return apply_slice(source, slices.ensure_normal_or_frozen, None)


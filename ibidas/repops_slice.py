from utils.multi_visitor import VisitorFactory, NF_ELSE
import repops

_delay_import_(globals(),"slices")
_delay_import_(globals(), "dim_helpers")
_delay_import_(globals(),"utils","util","context")
_delay_import_(globals(),"itypes","rtypes")

def realiasSimilarSlices(rsource, lall_slices, #{{{
                            keep_equal=False, return_newid=False,
                            always_disimilar_ids=set()):
    """Realiases similar slices, i.e. slices in rsource which 
    are also in lall_slices.

    Parameters
    ----------
    rsource: source of which the fields need to be checked and realised
    lall_slices: dictionary with slices from source against which to compare
    keep_equal: (Optional) only realias if slices have same id but are not equal (dif dims)
    return_newid: (Optional) also return new ids as a list
    always_disimilar_ids: (Optional) to be used in conjunction with keep_equal. Slices that
                          should be considered unequal. 

    Returns
    -------
    return_newid == False(default):  realised rsource
    returr_newid == True:            (realised rsource, realised ids)
    """

    re_id = []
    for slice in rsource._all_slices.values():  
        if(slice.id in lall_slices):
            if(slice.id in always_disimilar_ids):
                re_id.append(slice.id)
            elif(keep_equal and lall_slices[slice.id] == slice):
                continue
            else:
                re_id.append(slice.id)
    if(re_id):
        rsource = realias(rsource, re_id)
    if(return_newid):
        return (rsource, re_id)
    else:
        return rsource#}}}

class ProjectId(repops.UnaryOpRep):#{{{
    """Operation to select a number of slices, based on their id"""
    def __init__(self, source, selector):
        """
        Parameters
        ----------
        source: source to project upon
        selector: given to getActiveSlices to determine which slices to select
        """
        repops.UnaryOpRep.__init__(self, (source,), source._all_slices, 
                     self.getActiveSlices(source, selector))

    def getActiveSlices(self, source, sliceids):
        """Returns slices in source.all_slices (!) with id in sliceids"""
        return tuple([source._all_slices[slice_id] for slice_id in sliceids])
#}}}

@repops.delayable()
class project(VisitorFactory(prefixes=("project",), flags=NF_ELSE), #{{{
              ProjectId):
    """Project slices from active_slices that are indicated by 'selector'

    Parameters
    ----------
    source: source to project upon
    selector: can be name of slices, '*' for all active slices,
              tuple of the aforementioned options, idx of the active slice
              boolean index array, list of positions

    """
    def getActiveSlices(self, source, selector):
        return self.project(selector, source)

    def projectbasestring(self, selector, source):
        if(selector == "*"):
            return source._active_slices
        nactive_slices = [slice for slice in source._active_slices
                            if slice.name == selector]

        if(not nactive_slices):
            raise RuntimeError, "No slice: '" + str(selector) + "' found."
        elif(len(nactive_slices) > 1):  
            raise RuntimeError, "Multiple slices with name: '" + \
                                    str(selector) + "' found."
        return tuple(nactive_slices)

    def projecttuple(self, selector, source):
        return sum([self.project(elem, source) for elem in selector], ())

    def projectelse(self, selector, source):
        return util.select(source._active_slices, selector)
#}}}

def unpack_tuple(source, name="", unpack=True):#{{{
    """
    Parameters:
    source: source to unpack active slices from
    name: (Optional) if given, unpack only fields with this name
            if not given, unpack all fields from tuple.
    """
    assert len(source._active_slices) == 1, \
            "Unpacking tuples can only be done on single slices"
    slice = source._active_slices[0]
    if(not isinstance(slice.type, rtypes.TypeTuple)):
        if(name):
            raise RuntimeError, "Asked to unpack tuple attribute " + \
                name + " but cannot find a tuple."
        else:
            raise RuntimeError, "No tuple to unpack"

    if(not name):
        nactive_slices = [slices.UnpackTupleSlice(slice, idx)
                        for idx in range(len(slice.type.subtypes))]
    else: 
        try:
            idx = int(name)
        except ValueError:
            assert isinstance(name, str), \
                        "Tuple slice name should be a string"
            idx = slice.type.fieldnames.index(name)
        nactive_slices = [slices.UnpackTupleSlice(slice, idx)]
    
    all_slices = source._all_slices.copy()
    
    if(unpack):
        for pos, nslice in enumerate(nactive_slices):
            while(nslice.type.__class__ is rtypes.TypeArray):
                all_slices[nslice.id] = nslice
                nslice = slices.sunpack_array(nslice)
            nactive_slices[pos] = nslice
            all_slices[nslice.id] = nslice
    else:
        for nslice in nactive_slices:
            all_slices[nslice.id] = nslice

    self = repops.UnaryOpRep((source,), all_slices, 
            tuple(nactive_slices))
    return self#}}}

def slice_rename(source, *names, **kwds): #{{{
    nall_slices = source._all_slices.copy()
    nactive_slices = []
    if(names):
        assert (len(names) == len(source._active_slices)), \
            "Number of new slice names does not match number of active slices"
        for slice, name in zip(source._active_slices, names):
            nslice = slice.copy(realias=True)
            nslice.setName(name)
            nactive_slices.append(nslice)
            nall_slices[nslice.id] = nslice
    else:
        for slice in source._active_slices:
            if(slice.name in kwds):
                nslice = slice.copy(realias=True)
                nslice.name = kwds[slice.name]
                nall_slices[nslice.id] = nslice
            else:
                nslice = slice
            nactive_slices.append(nslice)
            
    self = repops.UnaryOpRep((source,), nall_slices, 
                    tuple(nactive_slices))
    return self
    #}}}

def realias(source, re_id): #{{{
    new_slices = {}
    nall_slices = source._all_slices.copy()
    for rid in re_id:
        nslice = nall_slices[rid].copy(realias=True)
        new_slices[rid] = nslice
        del nall_slices[rid]
        nall_slices[nslice.id] = nslice

    nactive_slices = tuple([new_slices.get(slice.id, slice) 
                                for slice in source._active_slices])
    self = repops.UnaryOpRep((source,), nall_slices, 
            tuple(nactive_slices))
    return self#}}}

def combine(*sources, **kwds):#{{{
    nall_slices = {}
    nsources = []
    for source in sources:
        nsource = realiasSimilarSlices(source, nall_slices, keep_equal=True)
        nsources.append(nsource)
        nall_slices.update(nsource._all_slices)
    sources = tuple(nsources)
    nactive_slices = sum([source._active_slices for source in sources],())
    
    self = repops.MultiOpRep(sources, nall_slices, tuple(nactive_slices))
            
    return self#}}}

@repops.delayable()
def rtuple(source, to_python=False):#{{{
    cdim = dim_helpers.commonDimPath(source)

    nall_slices = source._all_slices.copy()
    new_slices = {}
    for slice in source._all_slices.values():
        oslice = slice
        while(len(slice.dims) > len(cdim)):
            if(not to_python):
                slice = slices.PackArraySlice("array", slice, None)
            else:
                slice = slices.PackListSlice("list", slice, None)
            nall_slices[slice.id] = slice
        new_slices[oslice.id] = slice
   
    active_slices = [new_slices[slice.id] 
                        for slice in source._active_slices]
    
    nslice = slices.PackTupleSlice(active_slices, to_python=to_python)
    nall_slices[nslice.id] = nslice

    #initialize object attributes
    self = repops.UnaryOpRep((source,), nall_slices, 
        (nslice,))
    return self#}}}

import representor

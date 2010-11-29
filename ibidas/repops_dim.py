from repops import *

_delay_import_(globals(),"slices")
_delay_import_(globals(),"utils","util")


def broadcastDims(outer_dims, inner_dims):
    idim_pos = 0
    broadcast_actions = []
    for odim in outer_dims:
        if(idim_pos == len(inner_dims) or odim != inner_dims[idim_pos]):
            broadcast_actions.append(1)
        elif(not odim.variable):
            broadcast_actions.append(1 | 2)
            idim_pos += 1
        else:
            #variable dim, should have been unpacked simultaneously
            if(not 1 in broadcast_actions):
                broadcast_actions.append(1 | 2)
                idim_pos += 1
            else:
                break
    return (tuple(broadcast_actions), outer_dims[:len(broadcast_actions)], inner_dims[:idim_pos], 
                               outer_dims[len(broadcast_actions):], inner_dims[idim_pos:])


def matchDimSlices(all_slices, dims):
    res = {}
    
    for slice in all_slices.values():
        if(slice.dims[:len(dims)] == dims):
            res[slice.id] = slice
    return res
            
def redimSlices(all_slices, from_dims, to_dims, return_redim=False):
    res = {}
    copied = set()
    variables = any([dim.variable for dim in from_dims])

    if(not variables):
        for slice in all_slices.values():
            pos = util.contained_in(from_dims, slice.dims)
            if(not pos is False):
                slice = slice.copy()
                slice.dims = slice.dims[:pos[0]] + to_dims + slice.dims[pos[1]:]
                copied.add(slice)
            res[slice.id] = slice
    else:
        for slice in all_slices.values():
            if(slice.dims[:len(from_dims)] == from_dims):
                slice = slice.copy()
                slice.dims = to_dims + slice.dims[len(from_dims):]
                copied.add(slice)
            res[slice.id] = slice
            
    if(return_redim):
        return (res, copied)
    else:
        return res

def unpack_array(source, name = None):#{{{
    """Operation to unpack array typed slices

    Parameters
    ----------
    source: source with active slices which should be unpacked
    name: (Optional) name of dimension to unpack. If not given,  unpack all.
    """
    nactive_slices = []                       #new active slices
    
    for slice in source._active_slices:
        stype = slice.type

        #if name param, but does not match
        if(name and not (stype.dims and
                            any([name == dim.name for dim in stype.dims]))):
            nactive_slices.append(slice)
            continue
            
        nslice = slices.UnpackArraySlice(slice)
        nactive_slices.append(nslice)

    all_slices = source._all_slices.copy()
    for slice in nactive_slices:
        all_slices[slice.id] = slice

    #initialize object attributes
    self = UnaryOpRep((source,), all_slices, tuple(nactive_slices))
    return self#}}}
 
def redim(source, redim_map): #{{{
    for dimid_old, dim_new in redim_map.iteritems():
        if(dimid_old[-1].shape >=0 and dim_new.shape >= 0):
            assert dimid_old[-1].shape == dim_new.shape, \
                "Dimensions of slices do not match!"
    
    nall_slices = redimSlices2(source._all_slices, redim_map)

    nactive_slices = tuple([nall_slices[slice.id]
                            for slice in source._active_slices])
    self = UnaryOpRep((source,), nall_slices, 
            tuple(nactive_slices))
    return self#}}}

class dim_rename(UnaryOpRep):#{{{
    def __init__(self, source, *name, **kwds):
        nall_slices = source._all_slices.copy()
        nactive_slices = []
        for slice in source._active_slices:
            pass 
            
        pass
                
        UnaryOpRep.__init__(self, (source,), nall_slices, 
                     tuple(nactive_slices))#}}}

@delayable()
def rarray(source, dim=None):
    return apply_slice(source, "array", slices.PackArraySlice, dim)

@delayable()
def rlist(source, dim=None):
    return apply_slice(source, "list", slices.PackListSlice, dim)


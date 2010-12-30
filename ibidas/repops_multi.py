import operator

from constants import *
import repops

_delay_import_(globals(),"utils","util")
_delay_import_(globals(),"slices")
_delay_import_(globals(),"representor")
_delay_import_(globals(),"wrappers","wrapper_py")
_delay_import_(globals(),"itypes","rtypes","dimensions","dimpaths")
_delay_import_(globals(),"repops_funcs")

class Broadcast(repops.MultiOpRep):
    def process(self,sources, mode="dim"):
        state = reduce(operator.__and__,[source._state for source in sources])
        if not state & RS_SLICES_KNOWN:
            return

        nslices = []
        for bcslices in util.zip_broadcast(*[source._slices for source in sources]):
            nslices.extend(slices.broadcast(bcslices,mode))
        return self.initialize(tuple(nslices),state)


class Combine(repops.MultiOpRep):
    def process(self,sources):
        state = reduce(operator.__and__,[source._state for source in sources])
        if not state & RS_SLICES_KNOWN:
            return

        nslices = sum([source._slices for source in sources],tuple())
        return self.initialize(nslices,state)


class Filter(repops.MultiOpRep):
    def __init__(self,source,constraint,dim=None):
        if(not isinstance(constraint,representor.Representor)):
            constraint = repops.PlusPrefix(wrapper_py.rep(constraint,name="filter"))
        repops.MultiOpRep.__init__(self,(source,constraint),dim=dim)

    def process(self,sources,dim):
        source,constraint = sources
        if not source._state & RS_SLICES_KNOWN:
            return
        if not constraint._state & RS_TYPES_KNOWN:
            return

        assert len(constraint._slices) == 1, "Filter constraint should have 1 slice"
        cslice = constraint._slices[0]

        dimset = dimpaths.createDimSet([slice.dims for slice in source._slices])

        if(isinstance(cslice.type,rtypes.TypeBool)):
            assert dim is None, "Cannot use bool or missing data type with specified filter dimension. Constraint dimension already specifies dimension."
            assert cslice.dims, "Constraint should have at least one dimension"
            ndim = dimensions.Dim(UNDEFINED,tuple(),  False, name = "f" + cslice.dims[-1].name)
            seldim = cslice.dims[-1]
            assert seldim in dimset, "Cannot find last dimension of boolean filter in filter source (" + str(cslice.dims) + ")"
            cslice = slices.PackArraySlice(cslice,1)
        elif(isinstance(cslice.type, rtypes.TypeInteger)):
            dim_suffix = dimpaths.identifyDimPath([s.dims for s in source._slices], dim)
            seldim = dim_suffix[-1]
            ndim = None
        elif(isinstance(cslice.type, rtypes.TypeArray)):
            assert len(cslice.type.dims) == 1, "Filter array should be 1-dimensional"
            assert isinstance(cslice.type.subtypes[0], rtypes.TypeInteger) and \
                        not isinstance(cslice.type.subtypes[0], rtypes.TypeBool), \
                        "Multi-dimensional arrays cannot be used as filter. Please unpack the arrays."
            dim_suffix = dimpaths.identifyDimPath([s.dims for s in source._slices], dim)
            seldim = dim_suffix[-1]
            ndim = cslice.type.dims[0]
        elif(isinstance(cslice.type, rtypes.TypeSlice)):
            dim_suffix = dimpaths.identifyDimPath([s.dims for s in source._slices], dim)
            seldim = dim_suffix[-1]
            ndim = dimensions.Dim(UNDEFINED, seldim.dependent, False,name = "f" + cslice.name)
        else:
            raise RuntimeError, "Unknown constraint type in filter: " + str(cslice.type)

        if(isinstance(constraint, repops.PlusPrefix)):
            mode = "pos"
        else:
            mode = "dim"

        nslices = []
        for slice in source._slices:
            slice = slices.filter(slice, cslice, seldim, ndim, mode)
            nslices.append(slice)
        return self.initialize(tuple(nslices),source._state)
                    


def sort(source, sortsource):
    constraint = repops_funcs.ArgSort(sortsource)
    assert len(constraint._slices) == 1, "Sort field should have 1 slice"
    cslice = constraint._slices[0]
    return Filter(source, constraint.array(), dim=cslice.dims[-1])



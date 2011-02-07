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
    def _process(self,sources, mode="dim"):
        state = reduce(operator.__and__,[source._state for source in sources])
        if not state & RS_SLICES_KNOWN:
            return

        nslices = []
        for bcslices in util.zip_broadcast(*[source._slices for source in sources]):
            nslices.extend(slices.broadcast(bcslices,mode)[0])
        return self._initialize(tuple(nslices),state)


class Nest(repops.MultiOpRep):
    def __init__(self, lsource, rsource, dim=None):
        repops.MultiOpRep.__init__(self,(lsource,rsource),dim=dim)
    def _process(self, sources, dim=None):
        assert len(sources) == 2, "Nest expects two representor objects"
        state = reduce(operator.__and__,[source._state for source in sources])
        if not state & RS_SLICES_KNOWN:
            return
        lsource,rsource = sources
        
        joinpath = dimpaths.identifyUniqueDimPathSource(lsource, dim)
        joinpath = dimpaths.extendParentDim(joinpath,[s.dims for s in source._slices], ALL)

        idims = []
        for i in xrange(len(joinpath)):
            idims.append(dimensions.Dim(1))
        
        
        references = []
        for ndim in joinpath:
            nrefs = []
            for slice in lsource._slices:
                if ndim in slice.dims:
                    nrefs.append(slice)
            references.append(nrefs)

        nslices = []
        plan = [BCEXIST] * len(idims)
        for slice in rsource._slices:
            odims = slice.dims
            for dimpos in xrange(len(joinpath)):
                slice = slices.InsertDimSlice(slice,dimpos,idims[dimpos])
            slice = slices.BroadcastSlice(slice,references,plan,joinpath + odims)
            nslices.append(slice)
        return self._initialize(tuple(nslices), state)

class Combine(repops.MultiOpRep):
    def _process(self,sources):
        state = reduce(operator.__and__,[source._state for source in sources])
        if not state & RS_SLICES_KNOWN:
            return

        nslices = sum([source._slices for source in sources],tuple())
        return self._initialize(nslices,state)


class Filter(repops.MultiOpRep):
    def __init__(self,source,constraint,dim=None):
        if(not isinstance(constraint,representor.Representor)):
            constraint = repops.PlusPrefix(wrapper_py.rep(constraint,name="filter"))
        repops.MultiOpRep.__init__(self,(source,constraint),dim=dim)

    def _process(self,sources,dim):
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
            ndim = dimensions.Dim(UNDEFINED,(True,) * (len(cslice.dims) -1),  False, name = "f" + cslice.dims[-1].name)
            dim_suffix = None
            seldimpath = cslice.dims[-1:]
            assert seldimpath[0] in dimset, "Cannot find last dimension of boolean filter in filter source (" + str(cslice.dims) + ")"
            cslice = slices.PackArraySlice(cslice,1)
        else:
            seldimpath = dimpaths.identifyUniqueDimPathSource(source, dim)
            if(not seldimpath):
                raise RuntimeError, "Attempting to perform filter on non-existing dimension"

            if(isinstance(cslice.type, rtypes.TypeInteger)):
                ndim = None
            elif(isinstance(cslice.type, rtypes.TypeArray)):
                assert len(cslice.type.dims) == 1, "Filter array should be 1-dimensional"
                assert isinstance(cslice.type.subtypes[0], rtypes.TypeInteger) and \
                            not isinstance(cslice.type.subtypes[0], rtypes.TypeBool), \
                            "Multi-dimensional arrays cannot be used as filter. Please unpack the arrays."
                ndim = cslice.type.dims[0]
            elif(isinstance(cslice.type, rtypes.TypeSlice)):
                ndim = dimensions.Dim(UNDEFINED, (True,) * len(cslice.dims), False,name = "f" + cslice.name)
            else:
                raise RuntimeError, "Unknown constraint type in filter: " + str(cslice.type)


        if(isinstance(constraint, repops.PlusPrefix)):
            mode = "pos"
        else:
            mode = "dim"

        nslices = []
        for slice in source._slices:
            slice = slices.filter(slice, cslice, seldimpath, ndim, mode)
            nslices.append(slice)
        return self._initialize(tuple(nslices),source._state)
                    

def sort(source, *sortsources, **kwargs):
    descend = kwargs.pop("descend",False)
    if len(sortsources) > 1:
        sortsource = Combine(*sortsources)
    elif len(sortsources) == 0:
        if len(source._slices) > 1:
            sortsource = source.tuple()
        else:
            sortsource = source
    else:
        sortsource = sortsources[0]
    if len(sortsource._slices) > 1:
        sortsource = sortsource.tuple()

    constraint = repops_funcs.ArgSort(sortsource, descend=descend)
    assert len(constraint._slices) == 1, "Sort field should have 1 slice"
    cslice = constraint._slices[0]
    return Filter(source, constraint.array(), dim=cslice.dims[-1])



class Match(repops.MultiOpRep):
    def __init__(self, lsource, rsource, lfield=None, rfield=None):
        repops.MultiOpRep.__init__(self,(lsource,rsource),lfield=lfield,rfield=rfield)

    def _process(self, sources, lfield, rfield):
        assert len(sources) == 2, "Match expects two representor objects"
        state = reduce(operator.__and__,[source._state for source in sources])
        if not state & RS_SLICES_KNOWN:
            return
        lsource,rsource = sources
        if(lfield is None and rfield is None):
            rnames = set([slice.name for slice in rsource._slices])
            r = [slice for slice in lsource._slices if slice.name in rnames and slice.dims]
            if(len(r) != 1):
                raise RuntimeError, "Cannot find unique similarly named field to match. Please specify."
            name = r.pop()
            lslices = [slice for slice in lsource._slices if slice.name == name]
            rslices = [slice for slice in rsource._slices if slice.name == name]
        elif(rfield is None):
            lslices = [slice for slice in lsource._slices if slice.name == lfield]
            rslices = [slice for slice in rsource._slices if slice.name == lfield]
        else:
            lslices = [slice for slice in lsource._slices if slice.name == lfield]
            rslices = [slice for slice in rsource._slices if slice.name == rfield]
            
        if(len(lslices) > 1 or len(rslices) > 1):
            raise RuntimeError, "Matching slices in name not unique. Please rename or specify other slices."
        lslice = lslices[0]
        rslice = rslices[0]

       


        joinpath = dimpaths.identifyUniqueDimPathSource(lsource, dim)

        idims = []
        for i in xrange(len(joinpath)):
            idims.append(dimensions.Dim(1))
        
        
        references = []
        for ndim in joinpath:
            nrefs = []
            for slice in lsource._slices:
                if ndim in slice.dims:
                    nrefs.append(slice)
            references.append(nrefs)

        nslices = []
        plan = [BCEXIST] * len(idims)
        for slice in rsource._slices:
            odims = slice.dims
            for dimpos in xrange(len(joinpath)):
                slice = slices.InsertDimSlice(slice,dimpos,idims[dimpos])
            slice = slices.BroadcastSlice(slice,references,plan,joinpath + odims)
            nslices.append(slice)
        return self._initialize(tuple(nslices), state)
      

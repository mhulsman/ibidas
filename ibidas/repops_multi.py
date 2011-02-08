import operator

from constants import *
import repops

_delay_import_(globals(),"utils","util","context")
_delay_import_(globals(),"slices")
_delay_import_(globals(),"representor")
_delay_import_(globals(),"wrappers","wrapper_py")
_delay_import_(globals(),"itypes","rtypes","dimensions","dimpaths")
_delay_import_(globals(),"repops_funcs")
_delay_import_(globals(),"repops_dim")

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
        nslices = self._apply(*[source._slices for source in sources])
        return self._initialize(nslices,state)

    @classmethod
    def _apply(cls, *xslicelists):
        if(len(xslicelists) == 2):
            lslices,rslices = xslicelists
            lslices = [slices.ChangeBookmarkSlice(lslice,add_bookmark="!L",update_auto_bookmarks="L") for lslice in lslices]
            rslices = [slices.ChangeBookmarkSlice(rslice,add_bookmark="!R",update_auto_bookmarks="R") for rslice in rslices]
            return tuple(lslices + rslices)
        else:
            return sum([tuple(xslices) for xslices in xslicelists],tuple())

class Group(repops.MultiOpRep):
    def __init__(self,source,constraint,flat={}):
        repops.MultiOpRep.__init__(self,(source,constraint),flat=flat)

    def _process(self,sources, flat):
        source, gsource = sources 
        if not source._state & RS_SLICES_KNOWN or not gsource._state & RS_TYPES_KNOWN:
            return

        gslices = [slices.ensure_frozen(slice) for slice in gsource._slices]
        gslices = slices.broadcast(gslices,mode="dim")[0]
        gslices = [slices.PackArraySlice(gslice,1) for gslice in gslices]

        gslice = slices.GroupIndexSlice(gslices)
        gslice = slices.UnpackArraySlice(gslice, len(gslices))

        nslices = Filter._apply(source._slices,gslice,gslices[0].type.dims[:1],"dim")
        return self._initialize(tuple(nslices),source._state)


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
        seldimpath = dimpaths.identifyUniqueDimPathSource(source, dim)
        if(not seldimpath):
            raise RuntimeError, "Attempting to perform filter on non-existing dimension"
        
        if(isinstance(constraint, repops.PlusPrefix)):
            mode = "pos"
        else:
            mode = "dim"

        nslices = self._apply(source._slices,cslice,seldimpath,mode)

        return self._initialize(tuple(nslices),source._state)

    @classmethod
    def _apply(cls,fslices,cslice,seldimpath, bcmode):
        if(isinstance(cslice.type,rtypes.TypeBool)):
            assert cslice.dims, "Constraint should have at least one dimension"
            ndim = dimensions.Dim(UNDEFINED,(True,) * (len(cslice.dims) -1),  False, name = "f" + cslice.dims[-1].name)
            dim_suffix = None
            seldimpath = cslice.dims[-1:]
            dimset = dimpaths.createDimSet([slice.dims for slice in fslices])
            assert seldimpath[0] in dimset, "Cannot find last dimension of boolean filter in filter source (" + str(cslice.dims) + ")"
            cslice = slices.PackArraySlice(cslice,1)
        else:
            assert seldimpath, "Filter dimpath is empty"

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
        
        nslices = []
        for slice in fslices:
            slice = slices.filter(slice, cslice, seldimpath, ndim, bcmode)
            nslices.append(slice)
        return nslices

class Sort(repops.MultiOpRep):
    def __init__(self,source,constraint=None,descend=False):
        if(constraint is None):
            repops.MultiOpRep.__init__(self,(source,),descend=descend)
        else:
            if(not isinstance(constraint,representor.Representor)):
                constraint = repops.PlusPrefix(wrapper_py.rep(constraint,name="filter"))
            repops.MultiOpRep.__init__(self,(source,constraint),descend=descend)

    def _process(self, sources, descend):
        if not any([s._state & RS_SLICES_KNOWN for s in sources]):
            return

        if(len(sources) == 1): #no explicit constraint, use data itself
            source = sources[0]
            if len(source._slices) > 1:
                constraint = source.tuple()
            else:
                constraint = source
        else:
            source,constraint = sources 

        if len(constraint._slices) > 1:
            constraint = constraint.tuple()
        
        #fixme: make it slice-only (remove rep)
        constraint = repops_funcs.ArgSort(constraint, descend=descend)
        cslice = slices.PackArraySlice(constraint._slices[0])
        nslices = Filter._apply(source._slices, cslice, cslice.type.dims[:1],"dim")
        return self._initialize(tuple(nslices),source._state & constraint._state)


class Join(repops.MultiOpRep):
    def __init__(self, lsource, rsource, constraint):
        if(isinstance(constraint,context.Context)):
            c = Combine((lsource,rsource))
            constraint = context._apply(constraint,c)
        repops.MultiOpRep.__init__(self,(lsource,rsource, constraint))

    def _process(self, sources):
        lsource, rsource, constraint = sources
        if not lsource._state & RS_SLICES_KNOWN or not rsource._state & RS_SLICES_KNOWN \
           or not constraint._state & RS_TYPES_KNOWN:
            return
        
        ldims = dimpaths.createDimSet([s.dims for s in lsource._slices])
        rdims = dimpaths.createDimSet([s.dims for s in rsource._slices])
        
        assert len(constraint._slices) == 1, "Constraint should have only one slice"
        cslice = constraint._slices[0]

        if not cslice.dims[-1] in rdims:
            rdims,ldims = ldims,rdims
            lsource,rsource = rsource,lsource
            reverse = True
            assert cslice.dims[-1] in rdims, "Last dimension of constraint should be in one of the sources"
        else:
            reverse = False
        rdimpath = dimpaths.DimPath(cslice.dims[-1])
        for dim in cslice.dims[::-1]:
            if dim in ldims:
                ldimpath = dimpaths.DimPath(dim)


                break
        else:
            raise RuntimeError, "No dimension found for one of the sources in constraint"

        leftpos = repops_funcs.Pos._apply([cslice], ldimpath)[0]
        rightpos = repops_funcs.Pos._apply([cslice], rdimpath)[0]
        
        filters = Filter._apply([leftpos,rightpos],cslice, None,"dim")
        leftflat,rightflat = repops_dim.Flat._apply(filters, filters[0].dims[-1:])

        leftflat = slices.PackArraySlice(leftflat)
        rightflat = slices.PackArraySlice(rightflat)

        lslices = Filter._apply(lsource._slices, leftflat, ldimpath, "dim")
        rslices = Filter._apply(rsource._slices, rightflat, rdimpath, "dim")

        if(reverse):
            nslices = Combine._apply(rslices,lslices)
        else:
            nslices = Combine._apply(lslices,rslices)
        return self._initialize(tuple(nslices),lsource._state & rsource._state)



        



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
      

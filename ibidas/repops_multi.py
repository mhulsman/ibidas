import operator

from constants import *
import repops

_delay_import_(globals(),"utils","util","context")
_delay_import_(globals(),"ops")
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
            nslices.extend(ops.broadcast(bcslices,mode)[0])
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
                slice = ops.InsertDimOp(slice,dimpos,idims[dimpos])
            slice = ops.BroadcastOp(slice,references,plan,joinpath + odims)
            nslices.append(slice)
        return self._initialize(tuple(nslices), state)

class Combine(repops.MultiOpRep):
    def __init__(self, *sources):
        repops.MultiOpRep.__init__(self, sources)
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
            lslices = [ops.ChangeBookmarkOp(lslice,add_bookmark="!L",update_auto_bookmarks="L") for lslice in lslices]
            rslices = [ops.ChangeBookmarkOp(rslice,add_bookmark="!R",update_auto_bookmarks="R") for rslice in rslices]
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

        gslices = [ops.ensure_frozen(slice) for slice in gsource._slices]
        gslices = ops.broadcast(gslices,mode="dim")[0]
        gslices = [ops.PackArrayOp(gslice,1) for gslice in gslices]

        gslice = ops.GroupIndexOp(gslices)
        gslice = ops.UnpackArrayOp(gslice, len(gslices))

        #determine which slices to flatten
        oslices = list(source._slices)
        nslices = [None] * len(oslices)
        xflat = {}
        #specified slices
        for key,values in flat.iteritems():
            if(isinstance(key,tuple)):
                keyslices = gsource.Get(*key)
            else:
                keyslices = gsource.Get(key)
            nkey = tuple([gsource._slices.index(keyslice) for keyslice in keyslices._slices])
             
            if(isinstance(values,tuple)):
                vslices = source.Get(*values)
            else:
                vslices = source.Get(values)
            
            pos = []
            for slice in vslices._slices:
                assert slice in oslices, "Cannot find slice specified in group flat argument"
                p = oslices.index(slice)
                pos.append(p)
                oslices[p] = None
            xflat[nkey] = set(pos)

        #group slices
        for pos, slice in enumerate(gsource._slices):
            p = oslices.index(slice)
            key = (pos,)
            if(key in xflat):
                xflat[key].add(p)
            else:
                xflat[key] = set([p])
            oslices[p] = None

        #filter remaining slices first
        pos = [p for p,s in enumerate(oslices) if not s is None]
        if(pos):
            fslices = [s for s in oslices if not s is None]
            fslices = Filter._apply(fslices,gslice,gslices[0].type.dims[:1],"dim")
            for p,fslice in zip(pos,fslices):
                nslices[p] = fslice

        #filter flattened slices 
        gidx = range(len(gsource._slices))
        gdims = gslice.dims[-len(gsource._slices):]
        firstelem = wrapper_py.Rep(0)._slices[0]
        for key, pos in xflat.iteritems():
            tslice = gslice
            for elem in gidx:
                if not elem in key:
                    selpath = dimpaths.DimPath(gdims[elem])
                    tslice = repops_funcs.Sum._apply([tslice],selpath)[0]
            tslice = ops.UnpackArrayOp(tslice,1)
            tslice = Filter._apply([tslice], firstelem,tslice.dims,"dim")[0]
            tslice = ops.PackArrayOp(tslice,1)
            pos = list(pos)
            fslices = [source._slices[p] for p in pos]
            fslices = Filter._apply(fslices,tslice,gslices[0].type.dims[:1],"dim")
            for p,fslice in zip(pos,fslices):
                nslices[p] = fslice
            
        return self._initialize(tuple(nslices),source._state)


class Filter(repops.MultiOpRep):
    def __init__(self,source,constraint,dim=None):
        if(not isinstance(constraint,representor.Representor)):
            constraint = repops.PlusPrefix(wrapper_py.Rep(constraint,name="filter"))
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
        if(not seldimpath and not isinstance(cslice.type, rtypes.TypeBool)):
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
            cslice = ops.PackArrayOp(cslice,1)
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
                ndim = dimensions.Dim(UNDEFINED, (True,) * len(cslice.dims), False,name = "f" + seldimpath.strip()[-1].name)
            else:
                raise RuntimeError, "Unknown constraint type in filter: " + str(cslice.type)
        
        nslices = []
        for slice in fslices:
            slice = ops.filter(slice, cslice, seldimpath, ndim, bcmode)
            nslices.append(slice)
        return nslices

class Sort(repops.MultiOpRep):
    def __init__(self,source,constraint=None,descend=False):
        if(constraint is None):
            repops.MultiOpRep.__init__(self,(source,),descend=descend)
        else:
            if(not isinstance(constraint,representor.Representor)):
                constraint = repops.PlusPrefix(wrapper_py.Rep(constraint,name="filter"))
            repops.MultiOpRep.__init__(self,(source,constraint),descend=descend)

    def _process(self, sources, descend):
        if not any([s._state & RS_SLICES_KNOWN for s in sources]):
            return

        if(len(sources) == 1): #no explicit constraint, use data itself
            source = sources[0]
            if len(source._slices) > 1:
                constraint = source.Tuple()
            else:
                constraint = source
        else:
            source,constraint = sources 

        if len(constraint._slices) > 1:
            constraint = constraint.Tuple()
        
        #fixme: make it slice-only (remove rep)
        constraint = repops_funcs.Argsort(constraint, descend=descend)
        cslice = ops.PackArrayOp(constraint._slices[0])
        nslices = Filter._apply(source._slices, cslice, cslice.type.dims[:1],"dim")
        return self._initialize(tuple(nslices),source._state & constraint._state)


class Unique(repops.MultiOpRep):
    def __init__(self,source,constraint=None,descend=False):
        if(constraint is None):
            repops.MultiOpRep.__init__(self,(source,),descend=descend)
        else:
            if(not isinstance(constraint,representor.Representor)):
                constraint = repops.PlusPrefix(wrapper_py.Rep(constraint,name="filter"))
            repops.MultiOpRep.__init__(self,(source,constraint),descend=descend)

    def _process(self, sources, descend):
        if not any([s._state & RS_SLICES_KNOWN for s in sources]):
            return

        if(len(sources) == 1): #no explicit constraint, use data itself
            source = sources[0]
            if len(source._slices) > 1:
                constraint = source.Tuple()
            else:
                constraint = source
        else:
            source,constraint = sources 

        if len(constraint._slices) > 1:
            constraint = constraint.Tuple()
        
        #fixme: make it slice-only (remove rep)
        dpath = dimpaths.identifyUniqueDimPathSource(constraint,None)
        uconstraint = repops_funcs.Argunique(constraint)
        cslice = uconstraint._slices[0]
        nslices = Filter._apply(source._slices, cslice, dpath[-1:],"dim")
        return self._initialize(tuple(nslices),source._state & constraint._state)


class Join(repops.MultiOpRep):
    def __init__(self, lsource, rsource, constraint):
        if(isinstance(constraint,context.Context)):
            c = Combine(lsource,rsource)
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

        leftflat = ops.PackArrayOp(leftflat)
        rightflat = ops.PackArrayOp(rightflat)

        lslices = Filter._apply(lsource._slices, leftflat, ldimpath, "dim")
        rslices = Filter._apply(rsource._slices, rightflat, rdimpath, "dim")

        if(reverse):
            nslices = Combine._apply(rslices,lslices)
        else:
            nslices = Combine._apply(lslices,rslices)
        return self._initialize(tuple(nslices),lsource._state & rsource._state)


class Match(repops.MultiOpRep):
    def __init__(self, lsource, rsource, lslice=None, rslice=None, jointype="inner"):
        assert jointype in set(["inner","left","right","full"]), "Jointype should be inner, left, right or full"
        if(isinstance(lslice,context.Context)):
            lslice = context._apply(lslice,lsource)
        if(isinstance(rslice,context.Context)):
            rslice = context._apply(rslice,rsource)
        repops.MultiOpRep.__init__(self,(lsource,rsource,lslice,rslice),jointype=jointype)

    def _process(self, sources, jointype):
        lsource, rsource, lslice,rslice = sources
        if not lsource._state & RS_SLICES_KNOWN or not rsource._state & RS_SLICES_KNOWN:
            return
         
        if(lslice is None):
            if(rslice is None):
                common_names = set(lsource.Names) & set(rsource.Names)
                if(len(common_names) != 1):
                    raise RuntimeError, "Cannot find a unique common named slice"
                name = common_names.pop()
                lslice = getattr(lsource,name)
                rslice = getattr(rsource,name)
            else:
                lslice = getattr(lsource,rslice.Names[0])
        elif(rslice is None):
            rslice = getattr(rsource,lslice.Names[0])
        assert len(rslice._slices) == 1, "rslice parameter in match should have only one slice"
        assert len(lslice._slices) == 1, "lslice parameter in match should have only one slice"

        self._sources = (lsource, rsource, lslice, rslice)
        if not lslice._state & RS_TYPES_KNOWN or not rslice._state & RS_TYPES_KNOWN:
            return
        
        lslice = lslice._slices[0]
        rslice = rslice._slices[0]

        lindex,rindex = ops.EquiJoinIndexOp(lslice,rslice, jointype=jointype).results

        lslices = list(lsource._slices)
        rslices = list(rsource._slices)
        if(lslice.name == rslice.name and jointype=="inner" and lslice in lslices and rslice in rslices):
            rslices.pop(rslices.index(rslice))

        lslices = Filter._apply(lslices, lindex, lslice.dims, "dim")
        rslices = Filter._apply(rslices, rindex, rslice.dims, "dim")


        nslices = Combine._apply(lslices,rslices)
        return self._initialize(tuple(nslices),lsource._state & rsource._state)


class Stack(repops.MultiOpRep):
    def __init__(self, *sources, **kwargs):
        repops.MultiOpRep.__init__(self,sources, **kwargs)

    def _process(self, sources, dim=None):
        state = reduce(operator.__and__,[source._state for source in sources])
        if not state & RS_SLICES_KNOWN:
            return

        slicelens = set([len(source._slices) for source in sources])
        assert len(slicelens) == 1, "Sources of stack should have same number of slices"
        nslice = slicelens.pop()

        seldimpaths = [] 
        for source in sources:
            seldimpaths.append(dimpaths.identifyUniqueDimPathSource(source, dim))

        nslices = []
        dimdepth = []
        ndim = None
        slicelists = [source._slices for source in sources]
        for slicecol in zip(*slicelists):
            packdepths = []
            ncol = []
            for slice,dpath in zip(slicecol, seldimpaths):
                lastpos = slice.dims.matchDimPath(dpath)
                packdepths.append(len(slice.dims) - lastpos[-1])
                ncol.append(ops.PackArrayOp(slice,packdepths[-1]))
            
            res = ncol[0]
            for slice in ncol[1:]:
                res = repops_funcs.Add._apply((res,slice),"pos")
            dimdepth = min(packdepths)
            res = ops.UnpackArrayOp(res,dimdepth)
            
            if(len(nslices) == 0):
                ndim = res.dims[-dimdepth]
            else:
                res = ops.ChangeDimOp(res, len(res.dims) - dimdepth, ndim)
            nslices.append(res)

        return self._initialize(tuple(nslices),RS_CHECK)



        

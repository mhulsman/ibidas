import operator

from constants import *
import repops
import repops_funcs
import os

_delay_import_(globals(),"utils","util","context","config")
_delay_import_(globals(),"ops")
_delay_import_(globals(),"representor")
_delay_import_(globals(),"wrappers","python")
_delay_import_(globals(),"itypes","rtypes","dimensions","dimpaths")
_delay_import_(globals(),"repops_dim")
_delay_import_(globals(),"repops_slice")

class Broadcast(repops.MultiOpRep):
    def _sprocess(self,sources, mode="dim"):
        nslices = []
        for bcslices in util.zip_broadcast(*[source._slices for source in sources]):
            nslices.extend(ops.broadcast(bcslices,mode)[0])
        return self._initialize(tuple(nslices))


class Nest(repops.MultiOpRep):
    def __init__(self, lsource, rsource, dim=LASTCOMMONDIM):
        repops.MultiOpRep.__init__(self,(lsource,rsource),dim=dim)
    
    def _sprocess(self, sources, dim=None):
        assert len(sources) == 2, "Nest expects two representor objects"
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
        return self._initialize(tuple(nslices))

class Combine(repops.MultiOpRep):
    def __init__(self, *sources):
        repops.MultiOpRep.__init__(self, sources)
    
    def _sprocess(self,sources):
        sources = repops_dim.makeDimNamesUnique(*sources)
        nslices = self._apply(*[source._slices for source in sources])
        return self._initialize(nslices)

    @classmethod
    def _apply(cls, *xslicelists):
        if len(xslicelists) > 1:
            lslices,rslices = (xslicelists[0], xslicelists[-1])
            lslices = [ops.ChangeBookmarkOp(lslice,add_bookmark="!L",update_auto_bookmarks="L") for lslice in lslices]
            rslices = [ops.ChangeBookmarkOp(rslice,add_bookmark="!R",update_auto_bookmarks="R") for rslice in rslices]
            return tuple(lslices) + sum([tuple(xslices) for xslices in xslicelists[1:-1]],tuple()) + tuple(rslices)
        else:
            return xslicelists[0]

class Group(repops.MultiOpRep):
    def __init__(self,source,constraint,flat={}):
        repops.MultiOpRep.__init__(self,(source,constraint),flat=flat)

    def _process(self,sources, flat):
        source, gsource = sources 
        if not source._slicesKnown() or not gsource._typesKnown():
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
             
            if(isinstance(values,(list,tuple))):
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
            try:
                p = oslices.index(slice)
            except ValueError:
                continue
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
        firstelem = python.Rep(0,name="data")._slices[0]
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
            
        return self._initialize(tuple(nslices))


class Filter(repops.MultiOpRep):
    def __init__(self,source,constraint,dim=LASTCOMMONDIM,mode=None):
        if(not isinstance(constraint,representor.Representor)):
            data = python.Rep(constraint, name='filter')
            dims = data.DimsUnique
            if dims:
                dimnames = util.uniqify_names([d.name for d in dims],exclude=set([d.name for d in source.DimsUnique]))
                data = data%tuple(dimnames)
            constraint = repops.PlusPrefix(data)

        repops.MultiOpRep.__init__(self,(source,constraint),dim=dim,mode=mode)

    def _process(self,sources,dim,mode):
        source,constraint = repops_dim.makeDimNamesUnique(*sources)
        if not source._slicesKnown() or not constraint._typesKnown():
            return
        
        assert len(constraint._slices) == 1, "Filter constraint should have 1 slice"
        cslice = constraint._slices[0]

        if(not isinstance(cslice.type, rtypes.TypeBool)):
            seldimpath = dimpaths.identifyUniqueDimPathSource(source, dim)
        else:
            seldimpath = dimpaths.identifyUniqueDimPathSource(source, cslice.dims[-1])
            
        if(not seldimpath and not isinstance(cslice.type, rtypes.TypeBool)):
            raise RuntimeError, "Attempting to perform filter on non-existing dimension"
        
        if mode is None:
            if(isinstance(constraint, repops.PlusPrefix)):
                mode = "pos"
            else:
                mode = "dim"
           
        nslices = self._apply(source._slices,cslice,seldimpath,mode)
        return self._initialize(tuple(nslices))

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
                constraint = repops.PlusPrefix(python.Rep(constraint,name="filter"))
            repops.MultiOpRep.__init__(self,(source,constraint),descend=descend)

    def _sprocess(self, sources, descend):
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
        return self._initialize(tuple(nslices))


class Unique(repops.MultiOpRep):
    def __init__(self,source,constraint=None,descend=False):
        if(constraint is None):
            repops.MultiOpRep.__init__(self,(source,),descend=descend)
        else:
            if(not isinstance(constraint,representor.Representor)):
                constraint = repops.PlusPrefix(python.Rep(constraint,name="filter"))
            repops.MultiOpRep.__init__(self,(source,constraint),descend=descend)

    def _process(self, sources, descend):
        if(len(sources) == 1): #no explicit constraint, use data itself
            source = sources[0]
            if len(source._slices) > 1:
                constraint = source.Tuple()
            else:
                constraint = source
        else:
            source,constraint = sources 
        if not source._slicesKnown() or not constraint._typesKnown():
            return

        if len(constraint._slices) > 1:
            constraint = constraint.Tuple()
        
        #fixme: make it slice-only (remove rep)
        dpath = dimpaths.identifyUniqueDimPathSource(constraint,LASTCOMMONDIM)
        uconstraint = repops_funcs.Argunique(constraint)
        cslice = uconstraint._slices[0]
        nslices = Filter._apply(source._slices, cslice, dpath[-1:],"dim")
        return self._initialize(tuple(nslices))


class Join(repops.MultiOpRep):
    def __init__(self, lsource, rsource, constraint):
        if(isinstance(constraint,context.Context)):
            c = Combine(lsource,rsource)
            constraint = context._apply(constraint,c)
        repops.MultiOpRep.__init__(self,(lsource,rsource, constraint))

    def _process(self, sources):
        lsource, rsource, constraint = sources
        if not lsource._slicesKnown() or not rsource._slicesKnown() or not constraint._typesKnown():
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
        
        filters = Filter._apply([leftpos,rightpos],cslice, LASTCOMMONDIM,"dim")
        leftflat,rightflat = repops_dim.Flat._apply(filters, filters[0].dims[-1:])

        leftflat = ops.PackArrayOp(leftflat)
        rightflat = ops.PackArrayOp(rightflat)

        lslices = Filter._apply(lsource._slices, leftflat, ldimpath, "dim")
        rslices = Filter._apply(rsource._slices, rightflat, rdimpath, "dim")

        if(reverse):
            nslices = Combine._apply(rslices,lslices)
        else:
            nslices = Combine._apply(lslices,rslices)
        return self._initialize(tuple(nslices))

class Match(repops.MultiOpRep):
    def __init__(self, lsource, rsource, lslice=None, rslice=None, jointype="inner", merge_same=False, mode="dim"):
        assert jointype in set(["inner","left","right","full"]), "Jointype should be inner, left, right or full"
        assert (not isinstance(lslice,representor.Representor)), "Representor objects not allowed as lslice. Use context, string or int to indicate slice in lsource"
        assert (not isinstance(rslice,representor.Representor)), "Representor objects not allowed as rslice. Use context, string or int to indicate slice in rsource"
        
        if rslice is None and lslice is None:
            lsource,rsource = repops_dim.makeDimNamesUnique(lsource, rsource) 
 
        if not rslice is None:
            rslice = rsource.Get(rslice)
        elif not lslice is None:
            rslice = rsource.Get(lslice)

        if not lslice is None:
            lslice = lsource.Get(lslice)
            
        repops.MultiOpRep.__init__(self,(lsource,rsource,lslice,rslice),jointype=jointype, merge_same=merge_same,mode=mode)

    def _process(self, sources, jointype, merge_same,mode):
        lsource, rsource, lslice, rslice = sources
        if not lsource._slicesKnown() or not rsource._slicesKnown():
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
        if not lslice._slicesKnown() or not rslice._slicesKnown():
            return
        assert len(rslice._slices) == 1, "rslice parameter in match should have only one slice"
        assert len(lslice._slices) == 1, "lslice parameter in match should have only one slice"
        
        assert merge_same in set([False, 'equi','all',True]) or isinstance(merge_same,tuple), 'merge_same should be "equi", "all"/True or a tuple of names, or a tuple of tuple of name pairs'

        self._sources = (lsource, rsource, lslice, rslice)
        if not lslice._typesKnown() or not rslice._typesKnown():
            return
        lslice = lslice._slices[0]
        rslice = rslice._slices[0]
        nslices = self._apply(lsource, rsource, lslice, rslice, jointype, merge_same, mode)
        return self._initialize(tuple(nslices))


    @classmethod
    def _apply(cls,lsource, rsource, lslice,rslice, jointype="inner", merge_same=False, mode="dim"):
        lslice = ops.ensure_frozen(lslice)
        rslice = ops.ensure_frozen(rslice)

        leftslice = ops.PackArrayOp(lslice)
        rightslice = ops.PackArrayOp(rslice)
       
        ((bleftslice,brightslice),(leftplan,rightplan))= ops.broadcast((leftslice,rightslice),mode=mode)

        lindex,rindex = ops.EquiJoinIndexOp(bleftslice,brightslice, jointype=jointype, mode=mode).results

        lslices = list(lsource._slices)
        rslices = list(rsource._slices)
        
        if isinstance(merge_same, tuple):
            nms = dict()
            for ms in merge_same:
                if not isinstance(ms,tuple):
                    nms[ms] = ms
                else:
                    nms[ms[0]] = ms[1]
            merge_same = nms

        collapse_equi = (merge_same == 'equi' or merge_same == 'all' or merge_same is True) or (isinstance(merge_same, dict) and lslice.name in merge_same and rslice.name in merge_same[lslice.name])

        if((lslice.name == rslice.name or collapse_equi) and lslice in lslices and rslice in rslices):
            if jointype== 'inner':                
                rslices.pop(rslices.index(rslice))
            elif collapse_equi:
                if jointype == 'left':
                    rslices.pop(rslices.index(rslice))
                elif jointype == 'right':
                    lslices.pop(lslices.index(lslice))
                else:
                    lidx = lslices.index(lslice)
                    ridx = rslices.index(rslice)

        lslices = [ops.broadcastParentsFromPlan(tslice, lslice.dims[-1:], leftplan, leftslice.dims, bleftslice.dims, [rightslice], True) for tslice in lslices]
        rslices = [ops.broadcastParentsFromPlan(tslice, rslice.dims[-1:], rightplan, rightslice.dims, brightslice.dims, [leftslice], True) for tslice in rslices]

        lslices = list(Filter._apply(lslices, lindex, lslice.dims[-1:], "dim"))
        rslices = list(Filter._apply(rslices, rindex, rslice.dims[-1:], "dim"))

        if collapse_equi and jointype == 'full':
            lslice = lslices[lidx]
            rslice = rslices[ridx]
            lslices[lidx] = repops_funcs.Merge._apply([lslice,rslice], "dim")
            del rslices[ridx]

        
        if merge_same == 'all' or merge_same is True:
            ms = dict()
            lnames = [lslice.name for lslice in lslices]
            for rslice in list(rslices):
                if not rslice.name in lnames:
                    continue
                ms[rslice.name] = rslice.name                    
            merge_same = ms

        if isinstance(merge_same, dict):
            lnames = [lslice.name for lslice in lslices]
            rnames = [rslice.name for rslice in rslices]
            for lname, rname in merge_same.iteritems():
                if lname in lnames and rname in rnames:
                    lpos = lnames.index(lname)
                    rpos = rnames.index(rname)
                    lslices[lpos] = repops_funcs.Merge._apply([lslices[lpos],rslices[rpos]], "dim")
                    del rslices[rpos]

        return Combine._apply(lslices,rslices)



class Blast(repops.MultiOpRep):
    def __init__(self, lsource, rsource, lslice=None, rslice=None, blast_type = None, folder = None, reciprocal = True, normalize = False, overwrite = False, blastopts='', mode="dim"):
        assert blast_type in set(["nucl","prot", None]), "blast_type should be nucl or prot"
        assert (not isinstance(lslice,representor.Representor)), "Representor objects not allowed as lslice. Use context, string or int to indicate slice in lsource"
        assert (not isinstance(rslice,representor.Representor)), "Representor objects not allowed as rslice. Use context, string or int to indicate slice in rsource"
        
        if rslice is None and lslice is None:
            lsource,rsource = repops_dim.makeDimNamesUnique(lsource, rsource) 

        repops.MultiOpRep.__init__(self,(lsource,rsource,lslice,rslice), blast_type=blast_type, folder=folder, reciprocal=reciprocal, normalize=normalize, overwrite=overwrite, blastopts=blastopts, mode=mode)

    def _process(self, sources, blast_type, folder, reciprocal, normalize, overwrite, blastopts, mode):
        lsource, rsource, lslice, rslice = sources
        if not lsource._slicesKnown() or not rsource._slicesKnown():
            return

        lslice = self.find_seqslice(lsource, lslice)
        rslice = self.find_seqslice(rsource, rslice)

        if not lslice._typesKnown() or not rslice._typesKnown():
            return

        assert lslice.Type == rslice.Type, "Can not perform BLAST on different types of sequences, convert one to the other and try, try again."
        if blast_type is None:
            blast_type = "nucl" if (isinstance(lslice.Type, rtypes.TypeDNASequence)) else "prot"
        assert (isinstance(lslice.Type , rtypes.TypeDNASequence) and blast_type == 'nucl' ) or (isinstance(lslice.Type , rtypes.TypeProteinSequence) and blast_type == 'prot' ), "Can not perform this BLAST with this sequence type."
        
        self._sources = (lsource, rsource, lslice, rslice)
        lslice = lslice._slices[0]
        rslice = rslice._slices[0]
        nslices = self._apply(lsource, rsource, lslice, rslice, blast_type, folder, reciprocal, normalize, overwrite, blastopts, mode)
        return self._initialize(tuple(nslices))


    @classmethod
    def _apply(cls,lsource, rsource, lslice, rslice, blast_type = None, folder = None, reciprocal = True, normalize = False, overwrite = False, blastopts='', mode="dim"):
        leftslice = ops.PackArrayOp(lslice)
        rightslice = ops.PackArrayOp(rslice)

        if folder is None:
          folder = os.path.expanduser(config.config["blast_dir"])
        if not os.path.exists(folder):
            os.makedirs(folder, 0755)
       
        ((bleftslice,brightslice),(leftplan,rightplan))= ops.broadcast((leftslice,rightslice),mode=mode)

        r = ops.BlastIndexOp(bleftslice,brightslice, blast_type=blast_type, folder=folder, reciprocal=reciprocal, normalize=normalize, overwrite=overwrite, blastopts=blastopts).results
        blastres = [ ops.UnpackArrayOp(s,1) for s in r[2:] ]
        lindex = r[0]
        rindex = r[1]

        lslices = list(lsource._slices)
        rslices = list(rsource._slices)
        
        lslices = [ops.broadcastParentsFromPlan(tslice, lslice.dims[-1:], leftplan, leftslice.dims, bleftslice.dims, [rightslice], True) for tslice in lslices]
        rslices = [ops.broadcastParentsFromPlan(tslice, rslice.dims[-1:], rightplan, rightslice.dims, brightslice.dims, [leftslice], True) for tslice in rslices]

        lslices = list(Filter._apply(lslices, lindex, lslice.dims[-1:], "dim"))
        rslices = list(Filter._apply(rslices, rindex, rslice.dims[-1:], "dim"))

        return Combine._apply(tuple(lslices),tuple(blastres),tuple(rslices))

    def find_seqslice(self, source, seqslice, blast_type=None):
        if seqslice is None:
            seqslice = [ i for i in xrange(len(source._slices)) if isinstance(source._slices[i].type, rtypes.TypeSequence) ];
        else:
            seqslice = set(source.Get(seqslice)._slices)
            seqslice = [ i for i in xrange(len(source._slices)) if source._slices[i] in seqslice ];

        if len(seqslice) > 1:
            seqslice = seqslice[-1];
            util.warning("More than one sequence slice, using slice '%s'." % source._slices[seqslice].name);
        if len(seqslice) == 0:
            seqslice = len(source._slices) - 1;
            util.warning("No sequence slice specified, using last slice: '%s'." % source._slices[seqslice].name);
        else:
            seqslice = seqslice[-1];
        ss = source.Get(seqslice);

        if not isinstance(ss.Type, rtypes.TypeSequence):
            if blast_type == 'nucl':
                ss = ss.Cast("DNA");
            elif blast_type == 'prot':
                ss = ss.Cast("protein");
            else:
                raise RuntimeError, "Selected slice not of type sequence type, pass blast_type parameter."
        return ss

class Replace(repops.MultiOpRep):
    def __init__(self, source, slice, translator, fromslice=0, toslice=1):
        slice = source.Get(slice)
        fromslice = translator.Get(fromslice)
        toslice = translator.Get(toslice)
        translator = Combine(fromslice, toslice)

        repops.MultiOpRep.__init__(self,(source, slice, translator))
   
    def _sprocess(self, sources):
        source, slice, translator = sources
        assert len(slice._slices) == 1, "Only one slice can be replaced"
        assert len(translator._slices) == 2, "Translator should have only two selected slices"

        slice = slice._slices[0]
        slicepos = source._slices.index(slice)

        nslices = list(Match._apply(source, translator, slice, translator._slices[0],"inner"))
        nslices[slicepos] = ops.ChangeNameOp(nslices[-1],slice.name)
        remove = (len(source._slices) + len(translator._slices)) -len(nslices) - len(translator._slices)
        nslices = nslices[:remove]
        return self._initialize(tuple(nslices))


class Take(repops.MultiOpRep):
    def __init__(self, source, take_source, allow_missing=False, keep_missing=False):
        if (isinstance(source,dict)):
             source = python.Rep(list(source.iteritems()),name="data").IndexDict()
        repops.MultiOpRep.__init__(self,(source, take_source), allow_missing = allow_missing, keep_missing=keep_missing)
   
    def _sprocess(self, sources, allow_missing,keep_missing):
        source, take_source = sources
        if len(source._slices) == 2:
            source = source.IndexDict()
        assert len(source._slices) == 1, "Take source should have one slice"
        source_slice = source._slices[0]

        nslices = []
        for take_slice in take_source._slices:
            nslices.append(ops.TakeOp(source_slice, take_slice, allow_missing, keep_missing))
        return self._initialize(tuple(nslices))


class Stack(repops.MultiOpRep):
    def __init__(self, *sources, **kwargs):
        repops.MultiOpRep.__init__(self,sources, **kwargs)

    def _sprocess(self, sources, slices=COMMON_POS, mode='dim', dims=LCDIM):
        nslices = []
        if slices == COMMON_POS:
            slicelens = set([len(source._slices) for source in sources])
            assert len(slicelens) == 1, ("Sources should have same number of slices in %s when matching based on COMMON_POS" % self.__class__.__name__ )
        elif slices == COMMON_NAME:
            nms = set(sources[0].Names)
            assert len(sources[0].Names) == len(nms), ('Cannot have duplicate names in %s while matching slices on name' % self.__class__.__name__)
            for source in sources[1:]:
                xnms =  set(source.Names)
                assert len(source.Names) == len(xnms), ('Cannot have duplicate names in %s while matching slices on name' % self.__class__.__name__)
                nms &= xnms
            assert nms, ('Cannot find shared names in %s while matching slices on name' % self.__class__.__name__)

            nsources = []
            for source in sources:
                onms = set(source.Names) - nms
                if onms:
                    nslices.extend(source.Get(*onms)._slices)
                nsources.append(source.Get(*nms))
            sources = nsources
        
        seldimpaths = [] 
        if not isinstance(dims,tuple):
            dims = (dims,) * len(sources)
        for source, dim  in zip(sources, dims):
            seldimpaths.append(dimpaths.identifyUniqueDimPathSource(source, dim))

        slicelists = [source._slices for source in sources]
        ndim = None
        for slicecol in zip(*slicelists):
            packdepths = []
            ncol = []
            for slice,dpath in zip(slicecol, seldimpaths):
                lastpos = slice.dims.matchDimPath(dpath)
                assert len(lastpos) == 1, "Cannot choose between or find dims in slice: " + str(slice)
                packdepths.append(len(slice.dims) - lastpos[0])
                if(packdepths[-1] > 0):
                    slice = ops.PackArrayOp(slice, packdepths[-1])
                ncol.append(slice)
            
            #assert len(set([len(s.dims) for s in ncol])) == 1, "Dim depth mismatch between slices"

            res = ncol[0]
            for slice in ncol[1:]:
                res = repops_funcs.Add._apply((res,slice),mode)
            dimdepth = min(packdepths)
            res = ops.UnpackArrayOp(res,dimdepth)
            
            if ndim is None:
                ndim = res.dims[-dimdepth]
            else:
                res = ops.ChangeDimOp(res, len(res.dims) - dimdepth, ndim)
            nslices.append(res)

        return self._initialize(tuple(nslices))


class Intersect(repops.MultiOpRep):
    opercls = repops_funcs.And

    def __init__(self, *sources, **kwargs):
        repops.MultiOpRep.__init__(self,sources, **kwargs)

    def _sprocess(self, sources, slices=COMMON_POS, dims=LCDIM, mode='dim'):
        if any([not source._typesKnown() for source in sources]):
            return

        nslices = []
        if slices == COMMON_POS:
            slicelens = set([len(source._slices) for source in sources])
            assert len(slicelens) == 1, ("Sources should have same number of slices in %s when matching based on COMMON_POS" % self.__class__.__name__ )
        elif slices == COMMON_NAME:
            nms = set(sources[0].Names)
            assert len(sources[0].Names) == len(nms), ('Cannot have duplicate names in %s while matching slices on name' % self.__class__.__name__)
            for source in sources[1:]:
                xnms =  set(source.Names)
                assert len(source.Names) == len(xnms), ('Cannot have duplicate names in %s while matching slices on name' % self.__class__.__name__)
                nms &= xnms
            assert nms, ('Cannot find shared names in %s while matching slices on name' % self.__class__.__name__)

            nsources = []
            for source in sources:
                onms = set(source.Names) - nms
                if onms:
                    nslices.extend(source.Get(*onms)._slices)
                nsources.append(source.Get(*nms))
            sources = nsources
        elif isinstance(slices, tuple):
            assert len(slices) == len(sources), ('Slices tuple argument to %s should have same number of element as their are source datasets' % self.__class__.__name__)
            
            nsources = []
            for source, xslice in zip(sources, slices):
                if not isinstance(xslice, tuple):
                    xslice = (xslice,)
                selsource = source.Get(*xslice)                    
                noselsource = source.Without(selsource)
                nslices.extend(noselsource._slices)
                nsources.append(selsource)
            sources = nsources                
            
        seldimpaths = [] 
        tupleslices = []
        packdepthslist = []
        if not isinstance(dims,tuple):
            dims = (dims,) * len(sources)

        for dim, source in zip(dims, sources):
            dpath = dimpaths.identifyUniqueDimPathSource(source, dim)
            pslices = []
            packdepths = []
            for slice in source._slices:
                lastpos = slice.dims.matchDimPath(dpath)
                if not lastpos:
                    nslices.append(slice)
                else:
                    packdepths.append(len(slice.dims) - lastpos[-1] - 1)
                    assert len(lastpos) == 1, "Cannot find dims in slice: " + str(slice)
                    if(packdepths[-1] > 0):
                        slice = ops.PackArrayOp(slice, packdepths[-1])
                    pslices.append(slice)
            assert len(set([len(s.dims) for s in pslices])) == 1, "Dim depth mismatch between slices in a source"
            if(len(pslices) > 1):
                pslice = repops_slice.Tuple._apply(pslices)
            else:
                pslice = pslices[0]
            pslice = repops_funcs.Set._apply([pslice], None)[0]
            tupleslices.append(pslice)
            packdepthslist.append(packdepths)
        
        tupleslices = repops_dim.makeDimNamesUniqueSlices(*tupleslices) 
        res = tupleslices[0]
        for slice in tupleslices[1:]:
            res = self.opercls._apply((res,slice),mode)

        tslice = ops.UnpackArrayOp(res,1)
        if(isinstance(tslice.type, rtypes.TypeTuple)):
            pslices = [ops.UnpackTupleOp(tslice, idx) for idx in range(len(tslice.type.subtypes))]
        else:
            pslices = [tslice]
       
        for pos, pslice in enumerate(pslices):
            mpd = min([pd[pos] for pd in packdepthslist])
            if(mpd > 0):
                pslice = ops.UnpackArrayOp(pslice, mpd)
            nslices.append(pslice)

        return self._initialize(tuple(nslices))

class Union(Intersect):
    opercls = repops_funcs.Or


class Except(Intersect):
    opercls = repops_funcs.Subtract


class Difference(Intersect):
    opercls = repops_funcs.Xor



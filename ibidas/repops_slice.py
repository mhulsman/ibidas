import operator
from itertools import chain
from constants import *
import repops
import ops


_delay_import_(globals(),"representor")
_delay_import_(globals(),"wrappers","python")
_delay_import_(globals(),"utils","util","context")
_delay_import_(globals(),"itypes","rtypes","dimpaths","casts")

class ProjectDim(repops.UnaryOpRep):
    def _sprocess(self, source, name):
        nslices = [slice for slice in source._slices if slice.dims.hasName(name)]
        if not nslices:
            raise AttributeError, "Cannot find matching slices for dimension '" + str(name) + "'"
        self._initialize(tuple(nslices))

class ProjectBookmark(repops.UnaryOpRep):
    def _sprocess(self, source, name):
        nslices = [slice for slice in source._slices if name in slice.bookmarks]
        if not nslices:
            raise AttributeError, "Cannot find matching slices for bookmark '" + str(name) + "'"
        self._initialize(tuple(nslices))

class RequestUnaryOpRep(repops.UnaryOpRep):
    def __init__(self,source,*args, **kwds):
        self._req_sources = []
        repops.UnaryOpRep.__init__(self,source,*args, **kwds)


class To(RequestUnaryOpRep):
    def _sprocess(self, source, *slicesel, **kwargs):
        do = kwargs.pop("Do")
        assert not kwargs, "Unknown parameters: " + str(kwargs)

        if not self._req_sources:
            for ssel in slicesel:
                r = source.Get(ssel) 
                #assert len(r._slices) == 1, "To action can only be applied to single slices"
                all_pos = []
                for slice in r._slices:
                    #slice = r._slices[0]
                    assert slice in source._slices, "Selected slice in to should not be operated upon"
                    pos = source._slices.index(slice)
                    all_pos.append(pos)

                if(isinstance(do, context.Context)):
                    r = context._apply(do, r)
                else:
                    r = do(r)

                self._req_sources.append((all_pos, r))

        nslices = list(source._slices)
        for all_pos, r in self._req_sources:
            if not r._slicesKnown():
                return 
            for slice,pos in zip(r._slices, all_pos):
                nslices[pos] = slice

        self._initialize(tuple(nslices))
       

class Project(RequestUnaryOpRep):
    def __init__(self,source,*args, **kwds):
        self._req_sources = []
        repops.UnaryOpRep.__init__(self,source,*args, **kwds)

    def _getUsedSourceSlicesSet(self,req_sources):
        nslices = []
        for name, elem in req_sources:
            if isinstance(elem, representor.Representor):
                nslices.extend(elem._slices)
            else:
                nslices.extend(elem)
        return nslices

    def _sprocess(self, source, *args, **kwds):
        cur_slices = self._source._slices
        
        nslices = []
        if(not self._req_sources):
            req_sources = []
            for name, elem in chain(zip([None] * len(args),args),kwds.iteritems()):
                if(isinstance(elem, context.Context)):
                    elem = context._apply(elem, self._source)
                elif(isinstance(elem, basestring)):
                    if(elem == "~"):
                        used_slices = set([slice.name for slice in self._getUsedSourceSlicesSet(req_sources)])
                        nelem = [slice for slice in cur_slices if slice.name not in used_slices]
                    elif(elem == "#"):
                        common_dims = set([slice.dims for slice in cur_slices])
                        if len(common_dims) != 1:
                            raise RuntimeError, "Cannot use # selector as fields do not have a common dimension"
                        nelem = cur_slices[:1]
                    elif(elem == "*"):
                        nelem = cur_slices
                    else:
                        nelem = [slice for slice in cur_slices if slice.name == elem]
                        if(not nelem):
                            if len(cur_slices) == 1 and isinstance(cur_slices[0].type,rtypes.TypeTuple) \
                               and (elem in cur_slices[0].type.fieldnames or (isinstance(elem,int) and elem >= 0 and elem < len(cur_slices[0].type.fieldnames))):
                               nelem = UnpackTuple._apply(cur_slices[0],elem)
                            else:
                               nelem = [slice for slice in cur_slices if elem in slice.bookmarks]
                        
                        if not nelem:
                            raise AttributeError, "Cannot find attribute '" + elem + "'"
                    elem = nelem
                elif(isinstance(elem, representor.Representor)):
                    pass
                elif(isinstance(elem, tuple)):
                    elem = Tuple(self._source.Get(*elem))
                elif(isinstance(elem, list)):
                    if(len(elem) == 1):
                        elem = self._source.Get(*elem).Array()
                    else:
                        elem = self._source.Get(*elem).Array()
                else:
                    elem = util.select(cur_slices, elem)
                   
                req_sources.append((name,elem))
            self._req_sources = req_sources
     
        
        for name,elem in self._req_sources:
            if(isinstance(elem, representor.Representor)):
                if not elem._slicesKnown():
                    return 
                elem = [slice for slice in elem._slices]

            if(name):
                if not len(elem) == 1:
                    raise AttributeError, "Cannot find unique slice for '" + name + "', found: " + str(elem)
                nslices.append(ops.ChangeNameOp(elem[0],name))
            else:
                nslices.extend(elem)
        
        if not nslices:                
            raise AttributeError,  "No slices found with: " + str(args) + " and " + str(kwds)
        
        return self._initialize(tuple(nslices)) 
    

class Unproject(Project):
    def _sprocess(self, source, *args, **kwds):
        selslices = Project(source, *args, **kwds)._slices
        nslices = [slice for slice in source._slices if not slice in selslices]
        return self._initialize(tuple(nslices)) 


class AddSlice(repops.UnaryOpRep):
    def _sprocess(self, source, data, name=None, dtype=None):
        if not dtype is None:
            refdims = sum([slice.dims for slice in source._slices],tuple())
            dtype = rtypes.createType(dtype,refdims=refdims)
            if isinstance(data, representor.Representor):
                data = data()
            data = python.Rep(data, dtype=dtype, name=name)
        else: 
            if not isinstance(data, representor.Representor):
                data = python.Rep(data, name=name)
            else:
                data = data / name

        nslices = list(source._slices)
        nslices.extend(data._slices)
        return self._initialize(tuple(nslices)) 
       
class UnpackTuple(repops.UnaryOpRep):
    def _process(self,source,name=None,unpack=True):
        """
        Parameters:
        source: source to unpack active slices from
        name: (Optional) if given, unpack only fields with this name
                if not given, unpack all fields from tuple.
        """
        if not source._typesKnown():
            return

        if not len(source._slices) == 1:                
            if(name):
                raise AttributeError, "Asked to unpack tuple attribute '" + \
                    name + "', but cannot find a tuple."
            else:
                raise AttributeError, "No tuple to unpack"
        slice = source._slices[0]
        nslices = self._apply(slice,name,unpack=unpack)
        return self._initialize(tuple(nslices))

    @classmethod
    def _apply(cls, slice, name=None, unpack=True):
        if(not isinstance(slice.type, rtypes.TypeTuple)):
            if(name):
                raise AttributeError, "Asked to unpack tuple attribute '" + \
                    name + "', but cannot find a tuple"
            else:
                raise AttributeError, "No tuple to unpack"

        if(name is None):
            nslices = [ops.UnpackTupleOp(slice, idx) for idx in range(len(slice.type.subtypes))]
        else: 
            try:
                idx = int(name)
            except ValueError:
                assert isinstance(name, basestring), \
                            "Tuple slice name should be a string"
                try:
                    idx = slice.type.fieldnames.index(name)
                except ValueError:
                    raise AttributeError, "Cannot unpack, as tuple has no field '" + str(name) + "'"
            nslices = [ops.UnpackTupleOp(slice, idx)]
        
        if(unpack):
            for pos, nslice in enumerate(nslices):
                while(nslice.type.__class__ is rtypes.TypeArray):
                    nslice = ops.UnpackArrayOp(nslice)
                nslices[pos] = nslice
        return nslices

class Bookmark(repops.UnaryOpRep):
    def _sprocess(self, source, *names, **kwds): #{{{
        nslices = source._slices
        if(len(names) == 1):
            nslices = [ops.ChangeBookmarkOp(slice,names[0]) for slice in nslices]
        elif(len(names) > 1):
            unique_first_dims = util.unique([slice.dims[0] for slice in source._slices])
            assert (len(names) == len(unique_first_dims)), \
                "Number of new slice names does not match number of slices"
            for name,dim in zip(names,unique_first_dims):
                kwds[dim.name] = dim

        if(kwds):
            nnslices = []
            for slice in source._slices:
                if(slice.dims[0].name in kwds):
                    nslice = slice.ChangeBookmarkOp(slice,kwds[slice.dims[0].name])
                else:
                    nslice = slice
                nnslices.append(nslice)
            nslices = nnslices
                
        return self._initialize(tuple(nslices))
        #}}}


class SliceRename(repops.UnaryOpRep):
    def _sprocess(self, source, *names, **kwds): #{{{
        if(names):
            assert (len(names) == len(source._slices)), \
                "Number of new slice names does not match number of slices"
            nslices = [ops.ChangeNameOp(slice,name) 
                    for slice, name in zip(source._slices, names)]
        else:
            nslices = list(source._slices)
            for name, newname in kwds.iteritems():
                sels = [(pos,slice) for pos, slice in enumerate(source._slices) if slice.name == name]
                if isinstance(newname,tuple):
                    assert len(newname) == len(sels), 'Number of new names for slice %s does not match number of slices (%d)' % (name, len(sels))
                    for (pos,slice),nname in zip(sels,newname):
                        nslices[pos] = ops.ChangeNameOp(slice, nname)
                else:
                    for (pos,slice) in sels:
                        nslices[pos] = ops.ChangeNameOp(slice, newname)

        return self._initialize(tuple(nslices))
        #}}}

class SliceCast(repops.UnaryOpRep):
    def _process(self, source, *newtypes, **kwds): #{{{
        if not source._typesKnown():
            return
            
        if(newtypes):
            if len(newtypes) > 1:
                assert (len(newtypes) == len(source._slices)), \
                    "Number of new slice types does not match number of slices"
                nslices = [ops.CastOp(slice,rtypes.createType(newtype)) 
                        for slice, newtype in zip(source._slices, newtypes)]
            else:
                nslices = [ops.CastOp(slice,rtypes.createType(newtypes[0])) 
                        for slice in source._slices]
        else:
            nslices = list(source._slices)
            for name, newtype in kwds.iteritems():
                sels = [(pos,slice) for pos, slice in enumerate(source._slices) if slice.name == name]
                if isinstance(newtype,tuple):
                    assert len(newtype) == len(sels), 'Number of new types for slice %s does not match number of slices (%d)' % (name, len(sels))
                    for (pos,slice),ntype in zip(sels,newtype):
                        ntype = rtypes.createType(ntype)
                        nslices[pos] = ops.CastOp(slice, ntype)
                else:
                    for (pos,slice) in sels:
                        rnewtype = rtypes.createType(newtype)
                        nslices[pos] = ops.CastOp(slice, rnewtype)

        return self._initialize(tuple(nslices))
        #}}}


class Tuple(repops.UnaryOpRep):
    _ocls = ops.PackTupleOp
    def _sprocess(self, source, **kwargs):
        nslice = self._apply(source._slices, **kwargs)
        #initialize object attributes
        return self._initialize((nslice,))

    @classmethod
    def _apply(cls, slices, to_python=False, **kwargs):
        cdimpath = dimpaths.commonDimPath([slice.dims for slice in slices])
        nslices = []
        for slice in slices:
            oslice = slice
            if(to_python):
                while(len(slice.dims) > len(cdimpath)):
                    slice = ops.PackListOp(slice)
            else:
                if(len(slice.dims) > len(cdimpath)):
                    slice = ops.PackArrayOp(slice, ndim=len(slice.dims) - len(cdimpath))
            nslices.append(slice)
    
        return cls._ocls(nslices, **kwargs)
       
class Dict(Tuple):
    _ocls = ops.PackDictOp

class IndexDict(Tuple):
    _ocls = ops.PackIndexDictOp

class HArray(repops.UnaryOpRep):
    def _process(self, source, name=None):
        if not source._typesKnown():
            return
        
        #commonify dimensions
        cdimpath = dimpaths.commonDimPath([slice.dims for slice in source._slices])
        nslices = []
        for slice in source._slices:
            oslice = slice
            if(len(slice.dims) > len(cdimpath)):
                slice = ops.PackArrayOp(slice, ndim=len(slice.dims) - len(cdimpath))
            nslices.append(slice)

        #cast to common type
        ntype = casts.castMultipleImplicitCommonType(*[slice.type for slice in nslices])
        nnslices = []
        for slice in nslices:
            if(ntype != slice.type):
                slice = ops.CastOp(slice,ntype)
            nnslices.append(slice)
    
        if name is None:
            name = util.seq_names(1, exclude=set([d.name for d in source.DimsUnique]))[0]
    
        nslice = ops.HArrayOp(nnslices,name=name)
        nslice = ops.UnpackArrayOp(nslice)

        #initialize object attributes
        return self._initialize((nslice,))



class ToPythonRep(repops.UnaryOpRep):
    def _process(self, source):
        if not source._typesKnown():
            return

        nslices = []
        for slice in source._slices:
            if slice.type._requiresRPCconversion():
                slice = ops.ToPythonOp(slice)
            nslices.append(slice)
       
        if(len(nslices) > 1):
            nslice = Tuple._apply(nslices,to_python=True)
        else:
            nslice = nslices[0]

        while(nslice.dims):
            nslice = ops.PackListOp(nslice)

        return self._initialize((nslice,))



        

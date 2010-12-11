import operator
from itertools import chain
from constants import *
import repops


_delay_import_(globals(),"representor")
_delay_import_(globals(),"slices")
_delay_import_(globals(),"utils","util","context")
_delay_import_(globals(),"itypes","rtypes","dimpaths","casts")

class ProjectDim(repops.UnaryOpRep):
    def process(self, source, name):
        if not source._state & RS_SLICES_KNOWN:
            return
        
        nslices = [slice for slice in source._slices if slice.dims.hasName(name)]
        assert nslices,  "Cannot find matching slices for dimension: " + str(name)
        self.initialize(tuple(nslices),RS_CHECK)

class ProjectBookmark(repops.UnaryOpRep):
    def process(self, source, name):
        if not source._state & RS_SLICES_KNOWN:
            return
       
        nslices = [slice for slice in source._slices if name in slice.bookmarks]
        assert nslices,  "Cannot find matching slices for bookmark: " + str(name)
        self.initialize(tuple(nslices),RS_CHECK)

class Project(repops.UnaryOpRep):
    def _getUsedSourceSlicesSet(self,nslices):
        return nslices

    def process(self, source, *args, **kwds):
        if not source._state & RS_SLICES_KNOWN:
            return
        kwds = kwds.copy()
        check_one = kwds.pop("_check_one_",False)

        cur_slices = self._source._slices
        
        nslices = []
        for name, elem in chain(zip([None] * len(args),args),kwds.iteritems()):
            if(isinstance(elem, context.Context)):
                elem = context._apply(elem, self._source)
            elif(isinstance(elem,str)):
                if(elem == "~"):
                    used_slices = self._getUsedSourceSlicesSet(nslices)
                    nelem = [slice for slice in cur_slices if slice not in used_slices]
                elif(elem == "#"):
                    common_dims = set([slice.dims for slice in cur_slices])
                    if len(common_dims) != 1:
                        raise RuntimeError, "Cannot use # selector as fields do not have a common dimension"
                    nelem = cur_slices[:1]
                elif(elem == "*"):
                    nelem = cur_slices
                else:
                    nelem = [slice for slice in cur_slices if slice.name == elem]
                    if(not nelem and len(cur_slices) == 1 and isinstance(cur_slices[0].type,rtypes.TypeTuple)):
                        nelem = unpack_tuple(cur_slices[0],elem)
                    assert len(nelem) ==1, "Could not find (unique) matching slice for name: " + elem
                elem = nelem
            elif(isinstance(elem, Representor)):
                pass
            elif(isinstance(elem, tuple)):
                elem = RTuple(self._source.get(*elem))
            elif(isinstance(elem, list)):
                if(len(elem) == 1):
                    elem = self._source.get(*elem).array()
                else:
                    elem = self._source.get(*elem)
            else:
                elem = util.select(cur_slices, elem)

            if(isinstance(elem, representor.Representor)):
                if not elem._state & RS_SLICES_KNOWN:
                    return 
                elem = [slice for slice in elem._slices]

            if(name):
                assert len(elem) == 1, "Could not find a (unique) matching slice for " + name
                nslices.append(slices.ChangeNameSlice(elem,name))
            else:
                nslices.extend(elem)
        
        assert nslices, "No slices found with: " + str(args) + " and " + str(kwds)
        return self.initialize(tuple(nslices),self._source._state) 
    

class UnpackTuple(repops.UnaryOpRep):
    def process(self,source,name="",unpack=True):
        """
        Parameters:
        source: source to unpack active slices from
        name: (Optional) if given, unpack only fields with this name
                if not given, unpack all fields from tuple.
        """
        if not source._state & RS_TYPES_KNOWN:
            return

        assert len(source._slices) == 1, \
                "Unpacking tuples can only be done on single slices"
        slice = source._slices[0]
        nslices = unpack_tuple(slice,name,unpack)
        return self.initialize(tuple(nslices), RS_CHECK)
   
def unpack_tuple(slice,name="",unpack=True):
    if(not isinstance(slice.type, rtypes.TypeTuple)):
        if(name):
            raise RuntimeError, "Asked to unpack tuple attribute " + \
                name + " but cannot find a tuple."
        else:
            raise RuntimeError, "No tuple to unpack"

    if(not name):
        nslices = [slices.ensure_normal_or_frozen(slices.UnpackTupleSlice(slice, idx))
                                            for idx in range(len(slice.type.subtypes))]
    else: 
        try:
            idx = int(name)
        except ValueError:
            assert isinstance(name, str), \
                        "Tuple slice name should be a string"
            idx = slice.type.fieldnames.index(name)
        nslices = [slices.ensure_normal_or_frozen(slices.UnpackTupleSlice(slice, idx))]
    
    if(unpack):
        for pos, nslice in enumerate(nslices):
            while(nslice.type.__class__ is rtypes.TypeArray):
                nslice = slices.ensure_normal_or_frozen(slices.UnpackArraySlice(nslice))
            nslices[pos] = nslice
    return nslices

class Bookmark(repops.UnaryOpRep):
    def process(self, source, *names, **kwds): #{{{
        if not source._state & RS_SLICES_KNOWN:
            return
        
        nslices = source._slices
        if(len(names) == 1):
            nslices = [slices.ChangeBookmarkSlice(slice,names[0]) for slice in nslices]
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
                    nslice = slice.ChangeBookmarkSlice(slice,kdws[slice.dims[0].name])
                else:
                    nslice = slice
                nnslices.append(nslice)
            nslices = nnslices
                
        return self.initialize(tuple(nslices),source._state)
        #}}}


class SliceRename(repops.UnaryOpRep):
    def process(self, source, *names, **kwds): #{{{
        if not source._state & RS_SLICES_KNOWN:
            return
            
        if(names):
            assert (len(names) == len(source._slices)), \
                "Number of new slice names does not match number of slices"
            nslices = [slices.ChangeNameSlice(slice,name) 
                    for slice, name in zip(source._slices, names)]
        else:
            nslices = []
            for slice in source._slices:
                if(slice.name in kwds):
                    nslice = slice.ChangeNameSlice(slice,kdws[slice.name])
                else:
                    nslice = slice
                nslices.append(nslice)
                
        return self.initialize(tuple(nslices),source._state)
        #}}}

class SliceCast(repops.UnaryOpRep):
    def process(self, source, *newtypes, **kwds): #{{{
        if not source._state & RS_TYPES_KNOWN:
            return
            
        if(newtypes):
            assert (len(newtypes) == len(source._slices)), \
                "Number of new slice types does not match number of slices"
            nslices = [slices.CastSlice(slice,rtypes.createType(newtype)) 
                    for slice, newtype in zip(source._slices, newtypes)]
        else:
            nslices = []
            for slice in source._slices:
                if(slice.name in kwds):
                    newtype = rtypes.createType(kwds[slice.name])
                    nslice = slice.CastSlice(slice,newtype)
                else:
                    nslice = slice
                nslices.append(nslice)
                
        return self.initialize(tuple(nslices),source._state)
        #}}}

class Combine(repops.MultiOpRep):
    def process(self,*sources, **kwds):
        state = reduce(operator.__and__,[source._state for source in sources])
        if not state & RS_SLICES_KNOWN:
            return

        nslices = sum([source._slices for source in sources],tuple())
        return self.initialize(nslices,state)

@repops.delayable()
class RTuple(repops.UnaryOpRep):
    def process(self, source, to_python=False):
        if not source._state & RS_SLICES_KNOWN:
            return
        
        cdimpath = dimpaths.commonDimPath([slice.dims for slice in source._slices])
        nslices = []
        for slice in source._slices:
            oslice = slice
            if(to_python):
                while(len(slice.dims) > len(cdimpath)):
                    slice = slices.PackListSlice(slice)
            else:
                if(len(slice.dims) > len(cdimpath)):
                    slice = slices.PackArraySlice(slice, ndim=len(slice.dims) - len(cdimpath))
            nslices.append(slice)
    
        nslice = slices.PackTupleSlice(nslices, to_python=to_python)

        #initialize object attributes
        return self.initialize((nslice,),RS_ALL_KNOWN)

@repops.delayable()
class HArray(repops.UnaryOpRep):
    def process(self, source):
        if not source._state & RS_TYPES_KNOWN:
            return
        
        #commonify dimensions
        cdimpath = dimpaths.commonDimPath([slice.dims for slice in source._slices])
        nslices = []
        for slice in source._slices:
            oslice = slice
            if(len(slice.dims) > len(cdimpath)):
                slice = slices.PackArraySlice(slice, ndim=len(slice.dims) - len(cdimpath))
            nslices.append(slice)

        #cast to common type
        ntype = casts.castMultipleImplicitCommonType(*[slice.type for slice in nslices])
        nnslices = []
        for slice in nslices:
            if(ntype != slice.type):
                slice = slices.CastSlice(slice,ntype)
            nnslices.append(slice)
    
        nslice = slices.HArraySlice(nnslices)

        #initialize object attributes
        return self.initialize((nslice,),RS_ALL_KNOWN)

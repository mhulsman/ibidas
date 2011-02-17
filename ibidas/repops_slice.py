import operator
from itertools import chain
from constants import *
import repops
import ops


_delay_import_(globals(),"representor")
_delay_import_(globals(),"utils","util","context")
_delay_import_(globals(),"itypes","rtypes","dimpaths","casts")

class ProjectDim(repops.UnaryOpRep):
    def _process(self, source, name):
        if not source._state & RS_SLICES_KNOWN:
            return
        
        nslices = [slice for slice in source._slices if slice.dims.hasName(name)]
        assert nslices,  "Cannot find matching slices for dimension: " + str(name)
        self._initialize(tuple(nslices),RS_CHECK)

class ProjectBookmark(repops.UnaryOpRep):
    def _process(self, source, name):
        if not source._state & RS_SLICES_KNOWN:
            return
       
        nslices = [slice for slice in source._slices if name in slice.bookmarks]
        assert nslices,  "Cannot find matching slices for bookmark: " + str(name)
        self._initialize(tuple(nslices),RS_CHECK)


class To(repops.UnaryOpRep):
    def _process(self, source, *slicesel, **kwargs):
        if not source._state & RS_SLICES_KNOWN:
           return
        
        do = kwargs.pop("do")
        assert not kwargs, "Unknown parameters: " + str(kwargs)

        nslices = list(source._slices)
        for ssel in slicesel:
            r = source.get(ssel) 
            assert len(r._slices) == 1, "To action can only be applied to single slices"
            slice = r._slices[0]
            assert slice in source._slices, "Selected slice in to should not be operated upon"
            pos = source._slices.index(slice)

            if(isinstance(do, context.Context)):
                r = context._apply(do, r)
            else:
                r = do(r)

            nslices = nslices[:pos] + list(r._slices) + nslices[(pos+1):]

        self._initialize(tuple(nslices),RS_CHECK)
       

class Project(repops.UnaryOpRep):
    def __init__(self,source,*args, **kwds):
        self._project_sources = []
        repops.UnaryOpRep.__init__(self,source,*args, **kwds)

    def _getUsedSourceSlicesSet(self,project_sources):
        nslices = []
        for name, elem in project_sources:
            if isinstance(elem, representor.Representor):
                nslices.extend(elem._slices)
            else:
                nslices.extend(elem)
        return nslices

    def _process(self, source, *args, **kwds):
        if not source._state & RS_SLICES_KNOWN:
            return

        cur_slices = self._source._slices
        
        nslices = []
        if(not self._project_sources):
            project_sources = []
            for name, elem in chain(zip([None] * len(args),args),kwds.iteritems()):
                if(isinstance(elem, context.Context)):
                    elem = context._apply(elem, self._source)
                elif(isinstance(elem,str)):
                    if(elem == "~"):
                        used_slices = set([slice.name for slice in self._getUsedSourceSlicesSet(project_sources)])
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
                        if(not nelem and len(cur_slices) == 1 and isinstance(cur_slices[0].type,rtypes.TypeTuple) 
                            and (elem in cur_slices[0].type.fieldnames or (isinstance(elem,int) and elem >= 0 and elem < len(cur_slices[0].type.fieldnames)))):
                            nelem = unpack_tuple(cur_slices[0],elem)
                        assert len(nelem) ==1, "Could not find (unique) matching slice for name: " + elem
                    elem = nelem
                elif(isinstance(elem, representor.Representor)):
                    pass
                elif(isinstance(elem, tuple)):
                    elem = RTuple(self._source.get(*elem))
                elif(isinstance(elem, list)):
                    if(len(elem) == 1):
                        elem = self._source.get(*elem).array()
                    else:
                        elem = self._source.get(*elem).array()
                else:
                    elem = util.select(cur_slices, elem)

                project_sources.append((name,elem))
            self._project_sources = project_sources
     
        
        for name,elem in self._project_sources:
            if(isinstance(elem, representor.Representor)):
                if not elem._state & RS_SLICES_KNOWN:
                    return 
                elem = [slice for slice in elem._slices]

            if(name):
                assert len(elem) == 1, "Could not find a (unique) matching slice for " + name
                nslices.append(ops.ChangeNameOp(elem[0],name))
            else:
                nslices.extend(elem)
        assert nslices, "No slices found with: " + str(args) + " and " + str(kwds)
        
        if(any([s.type == rtypes.unknown for s in nslices])):
            state = RS_SLICES_KNOWN
        else:
            state = RS_ALL_KNOWN
        return self._initialize(tuple(nslices),state) 
    

class Unproject(Project):
    def _process(self, source, *args, **kwds):
        if not source._state & RS_SLICES_KNOWN:
            return

        selslices = Project(source, *args, **kwds)._slices
        nslices = [slice for slice in source._slices if not slice in selslices]
        return self._initialize(tuple(nslices),source._state) 




class UnpackTuple(repops.UnaryOpRep):
    def _process(self,source,name="",unpack=True):
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
        return self._initialize(tuple(nslices), RS_CHECK)
   
def unpack_tuple(slice,name="",unpack=True):
    if(not isinstance(slice.type, rtypes.TypeTuple)):
        if(name):
            raise RuntimeError, "Asked to unpack tuple attribute " + \
                name + " but cannot find a tuple."
        else:
            raise RuntimeError, "No tuple to unpack"

    if(not name):
        nslices = [ops.UnpackTupleOp(slice, idx) for idx in range(len(slice.type.subtypes))]
    else: 
        try:
            idx = int(name)
        except ValueError:
            assert isinstance(name, str), \
                        "Tuple slice name should be a string"
            idx = slice.type.fieldnames.index(name)
        nslices = [ops.UnpackTupleOp(slice, idx)]
    
    if(unpack):
        for pos, nslice in enumerate(nslices):
            while(nslice.type.__class__ is rtypes.TypeArray):
                nslice = ops.UnpackArrayOp(nslice)
            nslices[pos] = nslice
    return nslices

class Bookmark(repops.UnaryOpRep):
    def _process(self, source, *names, **kwds): #{{{
        if not source._state & RS_SLICES_KNOWN:
            return
        
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
                    nslice = slice.ChangeBookmarkOp(slice,kdws[slice.dims[0].name])
                else:
                    nslice = slice
                nnslices.append(nslice)
            nslices = nnslices
                
        return self._initialize(tuple(nslices),source._state)
        #}}}


class SliceRename(repops.UnaryOpRep):
    def _process(self, source, *names, **kwds): #{{{
        if not source._state & RS_SLICES_KNOWN:
            return
            
        if(names):
            assert (len(names) == len(source._slices)), \
                "Number of new slice names does not match number of slices"
            nslices = [ops.ChangeNameOp(slice,name) 
                    for slice, name in zip(source._slices, names)]
        else:
            nslices = []
            for slice in source._slices:
                if(slice.name in kwds):
                    nslice = slice.ChangeNameOp(slice,kdws[slice.name])
                else:
                    nslice = slice
                nslices.append(nslice)
                
        return self._initialize(tuple(nslices),source._state)
        #}}}

class SliceCast(repops.UnaryOpRep):
    def _process(self, source, *newtypes, **kwds): #{{{
        if not source._state & RS_TYPES_KNOWN:
            return
            
        if(newtypes):
            assert (len(newtypes) == len(source._slices)), \
                "Number of new slice types does not match number of slices"
            nslices = [ops.CastOp(slice,rtypes.createType(newtype)) 
                    for slice, newtype in zip(source._slices, newtypes)]
        else:
            nslices = []
            for slice in source._slices:
                if(slice.name in kwds):
                    newtype = rtypes.createType(kwds[slice.name])
                    nslice = slice.CastOp(slice,newtype)
                else:
                    nslice = slice
                nslices.append(nslice)
                
        return self._initialize(tuple(nslices),source._state)
        #}}}


class RTuple(repops.UnaryOpRep):
    _ocls = ops.PackTupleOp
    def _process(self, source):
        if not source._state & RS_SLICES_KNOWN:
            return
        
        nslice = self._apply(source._slices)
        #initialize object attributes
        return self._initialize((nslice,),RS_ALL_KNOWN)

    @classmethod
    def _apply(cls, slices, to_python=False):
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
    
        return cls._ocls(nslices)
       
class RDict(RTuple):
    _ocls = ops.PackDictOp

@repops.delayable()
class HArray(repops.UnaryOpRep):
    def _process(self, source, name=None):
        if not source._state & RS_TYPES_KNOWN:
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
    
        nslice = ops.HArrayOp(nnslices,name=name)

        #initialize object attributes
        return self._initialize((nslice,),RS_ALL_KNOWN)



class ToPythonRep(repops.UnaryOpRep):
    def _process(self, source):
        if not source._state & RS_TYPES_KNOWN:
            return

        nslices = []
        for slice in source._slices:
            if slice.type._requiresRPCconversion():
                slice = ops.ToPythonOp(slice)
            nslices.append(slice)
       
        if(len(nslices) > 1):
            nslice = RTuple._apply(nslices,to_python=True)
        else:
            nslice = nslices[0]

        while(nslice.dims):
            nslice = ops.PackListOp(nslice)

        return self._initialize((nslice,), RS_ALL_KNOWN)



        

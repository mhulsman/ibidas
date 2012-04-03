import math
import numpy
import operator
import collections
from ..constants import *
from ..utils.multi_visitor import VisitorFactory, NF_ERROR, NF_ELSE

_delay_import_(globals(), "..utils","sparse_arrays","cutils","missing","util")
_delay_import_(globals(), "..utils.missing","Missing")
_delay_import_(globals(), "rtypes")
_delay_import_(globals(), "dimpaths")

class BaseConvertor(object):
    def __init__(self,objectclss=set()):
        self.objectclss = objectclss

    def convert(self,seq,elem_type):
        return seq

class ArrayConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
        """Converts sequence to standard format. Converts None
        to Missing values."""
        return self._convert(dimpaths.getArrayDimPathFromType(elem_type),elem_type,seq)
 
    def _convert(self,dims,elem_type,seq):
        assert isinstance(elem_type,rtypes.TypeArray), "ArrayCOnvertor should be applied to array types"
        
        depth = dims.contigiousFixedNDims()
        if(not depth):# no fixed dims, allow for one non-fixed dim
            if(dims[0].has_missing):
                depth = 1  #no following dims (has_missing)
            else:
                depth = 1 + dims[1:].contigiousFixedNDims()
            variable=True
        else:
            variable=False

        rest_dims = elem_type.dims[depth:]
        if(rest_dims):
            elem_numpy_type = object
            elem_type = rtypes.TypeArray(elem_type.has_missing,dims=rest_dims,subtypes=elem_type.subtypes)
        else:
            elem_numpy_type = elem_type.subtypes[0].toNumpy()

        if(variable):
            seq_numpy_type = object
        else:
            seq_numpy_type = elem_numpy_type
            

        res = []
        for elem in seq:
            if(elem is Missing or elem is None):
                res.append(Missing)
                continue
            #convert to numpy array if needed
            if(not isinstance(elem, numpy.ndarray)):
                if(not isinstance(elem, collections.Sequence)):
                    elem = list(elem)
                elem = util.darray(elem,elem_numpy_type,depth,1)
            
            #assure correct number of dimensions
            if(len(elem.shape) < depth):
                oshape = elem.shape
                rem_ndims = depth - len(elem.shape) + 1
                elem = util.darray(list(elem.ravel()),elem_numpy_type,rem_ndims,rem_ndims)
                elem.shape = oshape + elem.shape[1:]
                assert len(elem.shape) == depth, "Non array values encountered for dims " + str(dims[len(elem.shape):])
            elif(len(elem.shape) > depth):
                oshape = elem.shape
                elem = dimpaths.flatFirstDims(elem,depth-1)
                elem = util.darray([subelem for subelem in elem],object,1,1)
                elem.shape = oshape[:depth]

            if not elem.dtype == elem_numpy_type:
                if(elem_numpy_type.char == 'S' or elem_numpy_type.char == 'U' or elem_numpy_type.char == 'V'):
                    z = numpy.zeros(elem.shape,elem_numpy_type)
                    z[:] = elem
                    elem = z
                else:
                    elem = numpy.cast[elem_numpy_type](elem)

            if(rest_dims):
                elem = self._convert(rest_dims,elem_type,elem)
            res.append(elem)

        if(variable):
            nseq = util.darray(res,seq_numpy_type)
        else:
            nseq = util.darray(res,seq_numpy_type,depth+1,1)

        return nseq

class StringConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
        #deduplicate
        dtype = elem_type.toNumpy()
        if(dtype == object):
            x = seq[:100]
            if(len(x) * 0.6 > len(set(x))):
                r = dict()
                res = []
                for elem in seq:
                    if(elem is Missing or elem is None):
                        res.append(Missing)
                    else:
                        try:
                            elem = r[elem]
                        except KeyError:
                            r[elem] = elem
                        res.append(elem)
                seq = util.darray(res)
        elif(not seq.dtype == dtype):
            #numpy bug: numpy.cast[<string dtype>] always cast to S1, irrespective 
            #requested length
            seq = util.darray(list(seq),numpy.string_)
        return seq


class FloatConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
        if(elem_type.has_missing):
            seqres = []
            for elem in seq:
                if(elem is Missing or elem == ""):
                    seqres.append(Missing)
                else: 
                    seqres.append(float(elem))
        else:
            seqres = [float(elem) for elem in seq]
         
        return util.darray(seqres,elem_type.toNumpy())

class IntegerConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
        if(elem_type.has_missing):
            seqres = []
            for elem in seq:
                if(elem is Missing or elem == ""):
                    seqres.append(Missing)
                else: 
                    seqres.append(int(elem))
        else:
            seqres = [int(elem) for elem in seq]
         
        return util.darray(seqres,elem_type.toNumpy())
 
class SetConvertor(BaseConvertor):

    def convert(self,seq,elem_type):
        if(not self.objectclss or set in self.objectclss): 
            if(elem_type.has_missing):
                seqres = []
                for elem in seq:
                    if(elem is None or elem is Missing):
                        seqres.append(Missing)
                    else: 
                        seqres.append(frozenset(elem))
            else:
                seqres = [frozenset(elem) for elem in seq]
        else:
            seqres = seq
        return util.darray(seqres,elem_type.toNumpy())



class DictConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
        columns = []
        for fieldname, subtype in zip(elem_type.fieldnames, elem_type.subtypes):
            def getname(elem):
                try:
                    return elem[fieldname]
                except (KeyError, TypeError):
                    return Missing
            
            subseq = util.darray([getname(elem) for elem in seq],subtype.toNumpy())
            columns.append(subseq)
        return util.darray(zip(*columns))

class NamedTupleConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
        columns = []
        for fieldname, subtype in zip(elem_type.fieldnames, elem_type.subtypes):
            def getname(elem):
                try:
                    return getattr(elem,fieldname)
                except (KeyError, TypeError):
                    return Missing
            subseq = util.darray([getname(elem) for elem in seq],subtype.toNumpy())
            columns.append(subseq)
        return util.darray(zip(*columns))

class TupleConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
        columns = []
        for pos, subtype in enumerate(elem_type.subtypes):
            itemfunc =  operator.itemgetter(pos)
            def getitem(elem):
                try:
                    return itemfunc(elem)
                except IndexError:
                    return Missing
            subseq = util.darray([getitem(elem) for elem in seq],subtype.toNumpy())
            columns.append(subseq)
        return util.darray(zip(*columns))





class slicetuple(tuple):
    def __new__(self,s):
        res = tuple.__new__(self,(s.start,s.stop,s.step))
        return res

class RPCConvertor(VisitorFactory(prefixes=("execConvert","fconvert"),flags=NF_ELSE)):
    def fconvertobject(self, obj):
        if(hasattr(obj, "__rpc_convert__")):
            return obj.__rpc_convert__()
        else:
            return str(obj)

    def _fnoconvert(self,obj):
        return obj
    fconvertbool = _fnoconvert
    fconvertint = _fnoconvert
    fconvertfloat = _fnoconvert
    fconvertstr = _fnoconvert
    fconvertbuffer = _fnoconvert

    def fconvertinteger(self, obj):
        return int(obj)

    def fconvertfloating(self,obj):
        return float(obj)
    
    def fconvertMissingType(self,obj):
        return None

    def fconvertstring_(self,obj):
        return str(obj)
    
    def fconvertunicode(self,obj):
        return unicode(obj)

    def fconvertbool_(self,obj):
        return bool(obj)

    def fconvertNone(self,obj):
        return None

    def fconverttuple(self, obj):
        return tuple([self.fconvert(elem) for elem in obj])

    def _fconvertsequence(self, obj):
        return [self.fconvert(elem) for elem in obj]
    fconvertlist = _fconvertsequence
    fconvertset = _fconvertsequence

    def fconvertndarray(self, obj):
        if(obj.dtype == object):
            return [self.fconvert(elem) for elem in obj]
        else:
            return obj.tolist()

    def fconvertslice(self, obj):
        return slicetuple(obj)
    
    def fconvertdict(self, obj):
        res = {}
        for key, value in obj.iteritems():
            res[self.fconvert(key)] = self.fconvert(value)
        return res

    def map(self, func, parent_type, seq):
        if(parent_type.has_missing):
            seqres = []
            for elem in seq:
                if(elem is Missing):
                    seqres.append(Missing)
                else: 
                    seqres.append(func(elem))
        else:
            seqres = [func(elem) for elem in seq]
        seqres=util.darray(seqres)
        return seqres 
    

    def execConvertTypeUnknown(self, ptype, seq):
        seq = self.map(self.fconvert, ptype, seq)
        return seq
    
    def execConvertTypeRecordDict(self, ptype, seq):
        if(ptype.has_missing):
            seqres = []
            for elem in seq:
                if elem is Missing:
                    r = Missing
                else:
                    r = dict([(key,self.fconvert(value)) for key,value in elem.iteritems()])
                seqres.append(r)
        else:
            seqres = [dict([(key, self.fconvert(value)) for key,value in elem.iteritems()]) for elem in seq]
        seqres = util.darray(seqres,seq.dtype)
        return seqres
  
    def execConvertTypeSlice(self, ptype, seq):
        return self.map(self.fconvertslice, ptype, seq)

    def execConvertTypeSet(self, ptype, seq):
        return self.map(self.fconvertset, ptype, seq)
   
rpc_convertor = RPCConvertor()   


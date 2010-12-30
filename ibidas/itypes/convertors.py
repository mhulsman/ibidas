import numpy
import collections
from ..constants import *

_delay_import_(globals(), "..utils","sparse_arrays","cutils","missing","util")
_delay_import_(globals(), "..utils.missing","Missing")
_delay_import_(globals(), "rtypes")
_delay_import_(globals(), "dimpaths")

def getConvertor(rtype):
    if 'convertor' in rtype.attr:
        return rtype.attr['convertor']
    else:
        return BaseConvertor(set([None.__class__,missing.MissingType]))

class BaseConvertor(object):
    def __init__(self,objectclss):
        self.objectclss = objectclss

    def convert(self,seq,elem_type):
        return seq

class ArrayConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
        """Converts sequence to standard format. Converts None
        to Missing values."""
        return self._convert(elem_type.getArrayDimPath(),elem_type,seq)
   
    def _convert(self,dims,elem_type,seq):
        assert isinstance(elem_type,rtypes.TypeArray), "ArrayCOnvertor should be applied to array types"
        cur_numpy_type = elem_type.toNumpy()
        
        depth = dims.contigiousFixedNDims()
        if(not depth):# no fixed dims, allow for one non-fixed dim
            if(dims[0].has_missing):
                depth = 1  #no following dims (has_missing)
            else:
                depth = 1 + dims[1:].contigiousFixedNDims()
            variable=True
        else:
            variable=False

        rest_dims = dims[depth:]
        if(rest_dims):
            elem_numpy_type = object
        else:
            elem_numpy_type = elem_type.getNestedArraySubtype().toNumpy()

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
                elem = cutils.darray(elem,object,depth,1)
            
            #assure correct number of dimensions
            if(len(elem.shape) < depth):
                oshape = elem.shape
                rem_ndims = depth - len(elem.shape) + 1
                elem = cutils.darray(list(elem.ravel()),elem_numpy_type,rem_ndims,rem_ndims)
                elem.shape = oshape + elem.shape[1:]
                assert len(elem.shape) == depth, "Non array values encountered for dims " + str(dims[len(elem.shape):])
            elif(len(elem.shape) > depth):
                oshape = elem.shape
                elem = dimpaths.flatFirstDims(elem,depth-1)
                elem = cutils.darray([subelem for subelem in elem],object,1,1)
                elem.shape = oshape[:depth]

            if(not isinstance(elem,sparse_arrays.FullSparse)):
                if not elem.dtype == elem_numpy_type:
                    elem = numpy.cast[elem_numpy_type](elem)
                elem = sparse_arrays.FullSparse(elem)

            if(rest_dims):
                elem = self._convert(rest_dims,elem_type,elem)
            res.append(elem)

        nseq = cutils.darray(res,seq_numpy_type,depth+1,1)
        seq = sparse_arrays.FullSparse(nseq)
        
        return seq

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
                nseq = cutils.darray(res)
                seq = sparse_arrays.FullSparse(nseq)
        elif(not seq.dtype == dtype):
            #numpy bug: numpy.cast[<string dtype>] always cast to S1, irrespective 
            #requested length
            seq = cutils.darray(list(seq),numpy.string_)
        return seq


class StringFloatConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
        nan = float("nan")
        if(elem_type.has_missing):
            seqres = []
            for elem in seq:
                if(not elem):
                    seqres.append(nan)
                else: 
                    seqres.append(float(elem))
        else:
            seqres = [float(elem) for elem in seq]
         
        nseq = cutils.darray(seqres,elem_type.toNumpy())
        seq = sparse_arrays.FullSparse(nseq)
        return seq

class StringIntegerConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
        if(elem_type.has_missing):
            seqres = []
            for elem in seq:
                if(not elem):
                    seqres.append(Missing)
                else: 
                    seqres.append(int(elem))
        else:
            seqres = [int(elem) for elem in seq]
         
        nseq = cutils.darray(seqres,elem_type.toNumpy())
        seq = sparse_arrays.FullSparse(nseq)
        return seq
 
class SetConvertor(BaseConvertor):

    def convert(self,seq,elem_type):
        if(set in self.objectclss): 
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
        nseq = cutils.darray(seqres,elem_type.toNumpy())
        seq = sparse_arrays.FullSparse(nseq)
        return seq



class DictConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
        for fieldname, subtype in zip(rtype.fieldnames, rtype.subtypes):
            def getname(elem):
                try:
                    return elem[fieldname]
                except (KeyError, TypeError):
                    return Missing
            
            subseq = cutils.darray([getname(elem) for elem in seq],subtype.toNumpy())
            columns.append(self.execFreeze(subtype, subseq))
        nseq = cutils.darray(zip(*columns))
        seq = sparse_arrays.FullSparse(nseq)
        return seq

class NamedTupleConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
        for fieldname, subtype in zip(rtype.fieldnames, rtype.subtypes):
            def getname(elem):
                try:
                    return getattr(elem,fieldname)
                except (KeyError, TypeError):
                    return Missing
            subseq = cutils.darray([getname(elem) for elem in seq],subtype.toNumpy())
            columns.append(self.execFreeze(subtype, subseq))
        nseq = cutils.darray(zip(*columns))
        seq = sparse_arrays.FullSparse(nseq)
        return seq


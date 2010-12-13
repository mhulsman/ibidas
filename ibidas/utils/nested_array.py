import numpy
import copy

_delay_import_(globals(),"..utils","cutils","sparse_arrays")
_delay_import_(globals(),"..utils.missing","Missing")
_delay_import_(globals(),"..itypes","dimpaths","rtypes")
class NestedArray(object):
    def __init__(self,data,cur_type):
        self.data = cutils.darray([data],object).view(sparse_arrays.FullSparse)
        self.cur_type = cur_type

        self.idxs = []
        self.idx_is_contigious = []
        self.dims = dimpaths.DimPath()
        
    def copy(self):
        res = copy.copy(self)
        res.data = numpy.array(self.data).view(sparse_arrays.FullSparse)
        res.idxs = list(self.idxs)
        res.idx_is_contigious = list(self.idx_is_contigious)
        return res
    
    def _curIdxDepth(self):
        if(not self.idxs or not isinstance(self.idxs[-1],int)):
            return 0
        else:
            return self.idxs[-1]+1
 
    def flat(self):
        data,rshape = self._flatData()
        data= cutils.darray([subelem for subelem in data],object,1,1)
        return data

    def _flatData(self):
        seq = self.data
        if(self.idxs and isinstance(self.idxs[-1],int)):
            rshape = seq.shape[:(self.idxs[-1]+2)]
            seq = dimpaths.flatFirstDims(seq,self.idxs[-1]+1)
        else:
            rshape = (len(seq),)
        #else:
        #    seq = cutils.darray([self.data],object).view(sparse_arrays.FullSparse)
        return (seq,rshape)

    def unpack(self, dimpath, subtype):
        nself = self.copy()
        #init
        data = nself.data
        depth = len(dimpath)
       
        while(depth>0):        
            idxdepth = nself._curIdxDepth()
            rem_dims = len(data.shape) - idxdepth - 1
            #does cur data have dimensions left that can be unpacked?
            if(rem_dims):
                rem_depth = min(rem_dims,depth)
                nself.idxs.extend(range(idxdepth, idxdepth + rem_depth))
                nself.idx_is_contigious.extend([True]*rem_depth)
                dimpath = dimpath[rem_depth:]
                depth -= rem_depth
                
                if(depth == 0): #if depth is 0 we can stop here
                    break
            
            #bummer, now it gets a bit more complicated
            cdata = data.ravel()
            #determine the number of fixed dimensions that can be unpacked
            #finds number of contiguous fixed dims from start of dimpath
            #This allows to unpack a multi-dimensional part of the data
            tot_dimpath = dimpath + subtype.getArrayDimPath()
            cdepth = tot_dimpath.contigiousFixedNDims()
            res = []
            
            #prepare fixed (multi-)dimensions and variable scenario
            if(not cdepth): 
                variable = True
                idxres = numpy.zeros((len(cdata),) + (2,),dtype=int)
                curpos = 0
                if(tot_dimpath[0].has_missing):
                    cdepth = 1
                else:
                    cdepth = 1 + tot_dimpath[1:].contigiousFixedNDims()
            else:
                variable = False

            for pos in xrange(len(cdata)):
                elem = cdata[pos]
                if(variable):
                    if(elem is Missing):
                        idxres[pos,:] = -1
                        continue
                    idxres[pos,0] = curpos
                    curpos = curpos + len(elem)
                    idxres[pos,1] = curpos
               
                #check that elem shape is not smaller or larger than expected 
                assert len(elem.shape) == cdepth, "Number of dimensions incorrect"
                res.append(elem)
            
            dimpath = dimpath[cdepth:]

            ndata = numpy.concatenate(res).view(sparse_arrays.FullSparse)
            if(variable):
                idxres.shape =  data.shape + (2,)
                nself.idxs.append(idxres)
            else:
                assert (ndata.shape[0] % len(cdata)) == 0, "Leftover elements in joining dimensions"
                pshape = ndata.shape[0] / len(cdata)
                ndata.shape = data.shape + (pshape,) + ndata.shape[1:]
                nself.idxs.append(len(data.shape)-1)
            nself.idx_is_contigious.append(True)
            data = ndata    
            depth -= cdepth
        
        nself.cur_type = subtype
        nself.dims = self.dims + dimpath
        nself.data = data
        return nself
    
    def pack(self, subtype, depth=1):
        #init
        nself = self.copy()
      
        while(depth):
            assert nself.idxs, "Pack operation on nestedarray without index?!"
            idx = nself.idxs.pop()
            nself.dims = nself.dims[:-1]
            nself.idx_is_contigious.pop()

            if(not isinstance(idx,int)): #refers to fixed dim in data
                res = []
                data = nself.data
                fidx = dimpaths.flatFirstDims(idx,len(idx.shape)-2)
                for pos in xrange(len(fidx)):
                     i = slice(fidx[pos,0],fidx[pos,1])
                     if(i.start == -1):
                        res.append(Missing)
                     else:
                        res.append(data[i])
                nself.data = cutils.darray(res,object).view(sparse_arrays.FullSparse)
                nself.data.shape = idx.shape[:-1]
            depth -= 1
        nself.cur_type = subtype
        return nself                
 
    def getStructuredData(self):
        return self.pack(rtypes.unknown, len(self.idxs)).data[0]

    def map(self, func, *args, **kwargs):
        restype= kwargs.get("res_type")
        dtype = restype.toNumpy()

        if(self.cur_type.has_missing):
            def wrapfunc(seq, *args, **kwargs):
                res = []
                for elem in seq:
                    if(elem is Missing):
                        res.append(Missing)
                    else:
                        res.append(func(elem, *args, **kwargs))
                nseq = cutils.darray(res,dtype)
                return nseq
        else:
            def wrapfunc(seq, *args, **kwargs):
                res = [func(elem, *args, **kwargs) for elem in seq]
                nseq = cutils.darray(res,dtype)
                return nseq
            
        return self.mapseq(wrapfunc, *args, **kwargs)

    def mapseq(self,func,*args,**kwargs):
        restype= kwargs.pop("res_type")
        dtype= restype.toNumpy()
        
        seq,rshape = self._flatData()
        seq = func(seq,*args,**kwargs)

        seq.shape = rshape + seq.shape[1:]
            
        if(not seq.dtype == dtype):
            seq = numpy.cast[dtype](seq)
            seq = seq.view(sparse_arrays.FullSparse)

        nself = self.copy()
        nself.data = seq
        nself.cur_type = restype
        return nself

    def insertDim(self,matchpoint,newdim):
        nself = self.copy()
        idxs = nself.idxs + [nself.data]
        curidx = idxs[matchpoint]
        if(isinstance(curidx,int)):
            newidx = curidx
        else:
            newidx = 0
        nself.idxs.insert(matchpoint,newidx)
        for pos in range(matchpoint+1,len(nself.idxs)):
            tidx = nself.idxs[pos]
            if(isinstance(tidx,int)):
                nself.idxs[pos] += 1
            else:
                tidx = numpy.array(tidx)
                tidx.shape = tidx.shape[:newidx] + (1,) + tidx.shape[newidx:]
                nself.idxs[pos] = tidx
                break
        else:
            nself.data.shape = nself.data.shape[:newidx] + (1,) + nself.data.shape[newidx:]
        nself.idx_is_contigious.insert(matchpoint,True)
        nself.dims = nself.dims[:matchpoint] + (newdim,) + nself.dims[matchpoint:]
        return nself

    def __repr__(self):
        return "NestedArray < \n" + \
               "Idxs: " + str(self.idxs) + "\n" + \
               "Data: " + str(self.data) + "\n" + \
               ">\n"



def co_mapseq(func, nested_arrays, *args, **kwargs):
    restype= kwargs.pop("res_type")
    dtype=restype.toNumpy()

    dimpath_set = set([na.dims for na in nested_arrays])
    assert len(dimpath_set) == 1, "Nested arrays should have same dimensions!"
    
    data = []
    for na in nested_arrays:
        seq,rshape = na._flatData()
        data.append(seq)
    
    seq = func(data,*args, **kwargs)
    seq.shape = rshape + seq.shape[1:]
    
    if(not seq.dtype == dtype):
        seq = numpy.cast[dtype](seq)
        seq = seq.view(sparse_arrays.FullSparse)

    nself = nested_arrays[0].copy()
    nself.data = seq
    nself.cur_type = restype
    return nself


def co_map(func, nested_arrays, *args, **kwargs):
    restype= kwargs.get("restype")
    dtype=restype.toNumpy()
    
    def wrapfunc(seqs, *args, **kwargs):
        res = [func(elems, *args, **kwargs) for elems in zip(*seqs)]
        nseq = cutils.darray(res,dtype)
        return nseq
    return self.co_mapseq(wrapfunc, *nested_arrays, **kwargs)

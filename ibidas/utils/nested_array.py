import numpy
import copy

_delay_import_(globals(),"..utils","cutils","sparse_arrays")
_delay_import_(globals(),"..utils.missing","Missing")
_delay_import_(globals(),"..itypes","dimpaths")
class NestedArray(object):
    def __init__(self,data):
        self.data = cutils.darray([data],object).view(sparse_arrays.FullSparse)
        self.idxs = []
        self.has_missing = [data is Missing]

    def copy(self):
        res = copy.copy(self)
        res.idxs = list(self.idxs)
        res.has_missing = list(self.has_missing)
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
                dimpath = dimpath[rem_depth:]
                depth -= rem_depth
                
                #set has missing flags
                nself.has_missing.extend([dim.has_missing for dim in dimpath[:rem_depth]])
                
                if(depth == 0): #if depth is 0 we can stop here
                    break
            
            #bummer, now it gets a bit more complicated
            cdata = data.ravel()
            #determine the number of fixed dimensions that can be unpacked
            #finds number of contiguous fixed dims from start of dimpath
            #This allows to unpack a multi-dimensional part of the data
            cdepth = (dimpath + subtype.getFullDimPath()).contigiousFixedNDims()
            res = []
            
            #prepare fixed (multi-)dimensions and variable scenario
            if(not cdepth): 
                variable = True
                idxres = numpy.zeros((len(cdata),) + (2,),dtype=int)
                curpos = 0
                cdepth = 1
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
            
            nself.has_missing.extend([dim.has_missing for dim in dimpath[:cdepth]])
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
            data = ndata    
            depth -= cdepth
        
        nself.has_missing.append(subtype.has_missing)
        nself.data = data
        return nself
    
    def pack(self, depth=1):
        #init
        nself = self.copy()
      
        while(depth):
            assert nself.idxs, "Pack operation on nestedarray without index?!"
            idx = nself.idxs.pop()
            has_missing = nself.has_missing.pop()
                
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
        return nself                
 
    def getStructuredData(self):
        return self.pack(len(self.idxs)).data[0]

    def map(self, func, *args, **kwargs):
        dtype= kwargs.get("dtype",object)
        if(self.has_missing[-1]):
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
        dtype= kwargs.pop("dtype",object)
        
        seq,rshape = self._flatData()
        seq = func(seq,*args,**kwargs)

        seq.shape = rshape + seq.shape[1:]
            
        if(not seq.dtype == dtype):
            seq = numpy.cast[dtype](seq)
            seq = seq.view(sparse_arrays.FullSparse)

        nself = self.copy()
        nself.data = seq
        return nself

    def normalize(self):
        nself = self.copy()
        all_data = nself.idxs + [nself.data]
        skip_dims = 0

        for pos in xrange(len(nself.idxs)):
            idx = all_data[pos]
            next_idx = all_data[pos + 1]
            if(isinstance(idx,int)):
                skip_dims = 0
                continue

            if(skip_dims):
                idx = dimpaths.flatFirstDims(dix,skip_dims)
                skip_dims = 0

            assert len(idx.shape) == 2, "Encountered index with incorrect dimensionality"

            #check 1: index array check for subsequent portions
            #          in the next index array
            temp_idx = idx[idx[:,0] != -1,0] #remove missing values
            difval = (temp_idx[1:,0] - temp_idx[:-1,0]).abs().sum()
            curpos = 0
            if(difval): #not subsequent...
                nidx = numpy.zeros(idx.shape,dtype=idx.dtype)
                next_newidx = []
                for row in xrange(len(idx)):
                    if(idx[row,0] == -1):
                        nidx[row,:] = -1
                    else:
                        k = slice(idx[row,0],idx[row,1])
                        nidx[row,0] = curpos
                        curpos += k.stop -  k.start
                        nidx[row,1] = curpos
                        next_newidx.append(next_idx[k])
                all_data[pos + 1] = numpy.concatenate(next_newidx)
                all_data[pos] = nidx
                next_idx = all_data[pos + 1]
                idx = nidx
            
            #check 2: check for fixed length
            temp_idx = idx[idx[:,0] != -1,0] #remove missing values
            lengths = numpy.unique(temp_idx[:,1] - temp_idx[:,0])
            if(len(lengths) == 1):
                nshape = lengths[0]
                if(len(temp_idx) == len(idx)):
                    assert (next_idx.shape[0] % nshape) == 0, "Incorrect first dimension of next index"
                    all_data[pos + 1].shape = (nshape,next_idx.shape[0] / nshape) + next_idx.shape[1:]
                else:
                    pass

                all_data[pos] = 1

            



                    
                
            
    
    def __repr__(self):
        return "NestedArray < \n" + \
               "Idxs: " + str(self.idxs) + "\n" + \
               "Data: " + str(self.data) + "\n" + \
               ">\n"



def co_mapseq(func, nested_arrays, *args, **kwargs):
    dtype= kwargs.pop("dtype",object)

    ndim = set([len(na.idxs) for na in nested_arrays])
    assert len(ndim) == 1, "Nested arrays should have same dimension!"
    ndim = ndim.pop()
    
    prim_na = nested_arrays[0]
    walk_necessary = -1
    for i in xrange(ndim):
        idxa = prim_na.idxs[i]
        for na in nested_arrays:
            idxb = na.idxs[i]
            if(isinstance(idxa,int) and isinstance(idxb,int) and idxa == idxb):
                continue
            elif(isinstance(idxa,numpy.ndarray) and isinstance(idxb, numpy.ndarray)
                and (idxa == idxb).all()):
                continue
            else:
                walk_necessary = i
                break
        if(walk_necessary >= 0):
            break

    if(walk_necessary >= 0):
        curposses = [0] * ndim
        max_length = [0] * ndim
        stack = [[None] * len(nested_arrays) for pos in range(ndim + 1)]
        idxpos = walk_necessary

        while(idxpos >= walk_necessary):
            if(idxpos < ndim):
                for npos,na in enumerate(nested_arrays):
                    idx = stack[idxpos][npos]
                    if(isinstance(idx,list)):
                        curfilter = idx
                    else:
                        curfilter = []
                    curidxpos = curposses[idxpos]
                    if(instance(idx,int)):
                        curfilter.append(idx)
                        if((idxpos + 1) == ndim):
                            curfilter.append(Ellipsis)
                            stack[idxpos+1][npos] = na.data[tuple(curfilter)]
                        else:
                            stack[idxpos+1][npos] = curfilter
                    else:
                        curslice = slice(idx[curidxpos,0],idx[curidxpos,1])
                        curfilter.append(curslice)
                        curfilter.append(Ellipsis)
                        if((idxpos + 1) == ndim):
                            stack[idxpos + 1][npos] = na.data[tuple(curfilter)]
                        else:
                            stack[idxpos + 1][npos] = na.idxs[curpos + 1][tuple(curfilter)]

                idxpos += 1
            else:
                seq = func(stack[-1],*args, **kwargs)
                #run mapseq
            
                

            
            
            

    else:
        seq = func([na.data for na in nested_arrays],*args, **kwargs)

    seq.shape = nested_arrays[0].data.shape
    if(not seq.dtype == dtype):
        seq = numpy.cast[dtype](seq)
        seq = seq.view(sparse_arrays.FullSparse)

    nself = nested_arrays[0].copy()
    nself.data = seq
    return nself


def co_map(func, nested_arrays, *args, **kwargs):
    dtype= kwargs.get("dtype",object)
    
    def wrapfunc(seqs, *args, **kwargs):
        res = [func(elems, *args, **kwargs) for elems in zip(*seqs)]
        nseq = cutils.darray(res,dtype)
        return nseq
    return self.co_mapseq(wrapfunc, *nested_arrays, **kwargs)

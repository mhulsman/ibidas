import numpy
import copy

_delay_import_(globals(),"..utils","cutils","sparse_arrays","util")
_delay_import_(globals(),"..utils.missing","Missing")
_delay_import_(globals(),"..itypes","dimpaths","rtypes","dimensions")
class NestedArray(object):
    def __init__(self,data,cur_type):
        self.data = cutils.darray([data],object)
        self.cur_type = cur_type

        self.idxs = [0]
        tdim = dimensions.Dim(1)
        self.dims = dimpaths.DimPath(tdim)
        
    def copy(self):
        res = copy.copy(self)
        res.data = numpy.array(self.data)
        res.idxs = list(self.idxs)
        return res
  
    def getDim(self,pos):
        pos += 1
        return self.dims[pos]

    def getDimShape(self,pos):
        pos += 1
        if(isinstance(self.idxs[pos],int)):
            nextpos,nextobj = self._get_next_obj(pos)
            return nextobj.shape[self.idxs[pos]]
        else:
            return self.idxs[pos]

    def _curIdxDepth(self):
        if(isinstance(self.idxs[-1],int)):
            return self.idxs[-1] + 1
        else:
            return 1

    def replaceDim(self,pos,ndim):
        pos += 1
        nself = self.copy()
        dims = list(self.dims)
        dims[pos] = ndim
        nself.dims = dimpaths.DimPath(*dims)
        return nself

    def flat(self):
        data,rshape = self._flatData()
        data= cutils.darray([subelem for subelem in data],object,1,1)
        return data

    def _flatData(self):
        seq = self.data
        if(isinstance(self.idxs[-1],int)):
            rshape = seq.shape[:(self.idxs[-1]+1)]
            seq = dimpaths.flatFirstDims(seq,self.idxs[-1])
        else:
            rshape = (len(seq),)
        return (seq,rshape)

    def unpack(self, dimpath, subtype):
        nself = self.copy()
        #init
        data = nself.data
        depth = len(dimpath)
        odimpath = dimpath
       
        while(depth>0):        
            idxdepth = nself._curIdxDepth()
            rem_dims = len(data.shape) - idxdepth
            #does cur data have dimensions left that can be unpacked?
            if(rem_dims):
                rem_depth = min(rem_dims,depth)
                nself.idxs.extend(range(idxdepth, idxdepth + rem_depth))
                dimpath = dimpath[rem_depth:]
                depth -= rem_depth
                
                if(depth == 0): #if depth is 0 we can stop here
                    break
            
            #bummer, now it gets a bit more complicated
            cdata = data.ravel()
            #determine the number of fixed dimensions that can be unpacked
            #finds number of contiguous fixed dims from start of dimpath
            #This allows to unpack a multi-dimensional part of the data
            tot_dimpath = dimpath + dimpaths.getArrayDimPathFromType(subtype)
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

            ndata = numpy.concatenate(res)
            if(variable):
                idxres.shape =  data.shape + (2,)
                nself.idxs.append(idxres)
            else:
                assert (ndata.shape[0] % len(cdata)) == 0, "Leftover elements in joining dimensions"
                pshape = ndata.shape[0] / len(cdata)
                ndata.shape = data.shape + (pshape,) + ndata.shape[1:]
                nself.idxs.append(len(data.shape))
            data = ndata    
            depth -= 1
        
        nself.cur_type = subtype
        nself.dims = self.dims + odimpath
        nself.data = data
        return nself
    
    def pack(self, subtype, depth=1):
        #init
        nself = self.copy()
      
        while(depth):
            assert len(nself.idxs) > 1, "Pack operation on nestedarray without index?!"
            assert len(nself.dims) == len(nself.idxs), "Idxs and dims not equal in size"
            idx = nself.idxs.pop()
            nself.dims = nself.dims[:-1]

            if(not isinstance(idx,int)): #refers to fixed dim in data
                res = []
                data = nself.data
                fidx = dimpaths.flatFirstDims(idx,len(idx.shape)-2)
                for pos in xrange(len(fidx)):
                     start = fidx[pos,0]
                     stop = fidx[pos,1]
                     if(start == -1):
                        res.append(Missing)
                     else:
                        res.append(data[start:stop])
                nself.data = cutils.darray(res,object)
                nself.data.shape = idx.shape[:-1]
            depth -= 1
        nself.cur_type = subtype
        return nself                
 
    def getStructuredData(self):
        return self.pack(rtypes.unknown, len(self.idxs)-1).data[0]

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
        
        seq,rshape = self._flatData()
        seq = func(seq,*args,**kwargs)
        seq.shape = rshape + seq.shape[1:]

        nself = self.copy()
        nself.data = seq
        nself.cur_type = restype
        return nself

    def insertDim(self,matchpoint,newdim):
        matchpoint += 1
        nself = self.copy()
        idxs = nself.idxs + [nself.data]
        curidx = idxs[matchpoint-1]
        if(isinstance(curidx,int)):
            newidx = curidx+1
        else:
            newidx = 1
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
        nself.dims = nself.dims[:matchpoint] + (newdim,) + nself.dims[matchpoint:]
        return nself

    def mergeAllDims(self, newdim):
        nself = self.copy()
        nself.idxs = [0,1]
        nself.dims = self.dims[:1] + (newdim,)
        nself.data = self._flatData()[0]
        nself.data.shape = (1,) + nself.data.shape
        return nself
        
    def mergeLastDims(self, ndims, newdim):
        nself = self.copy()
        nself.dims = self.dims[:-ndims] + (newdim,)
        nself.data = self._flatData()[0]
        nself.idxs = self.idxs[:-ndims]
        lastidx = nself.idxs[-1]
        if(isinstance(lastidx,int)):
            nself.idxs.append(nself.idxs[-1] + 1)
            nextpos, nextobj = self._get_next_obj(len(nself.idxs))
            nelem = numpy.multiply.reduce(nextobj.shape[:(lastidx + 1)])
            assert (nself.data.shape[0] % nelem) == 0, "Leftover elements in merging"
            nself.data.shape = nextobj.shape[:(lastidx + 1)] + (nself.data.shape[0] / nelem) + nself.data.shape[1:]
        else:
            nself.idxs.append(0)
            lastidx = lastidx.ravel()
            nelem = lastidx[-1] - lastidx[0]
            assert (nself.data.shape[0] % nelem) == 0, "Leftover elements in merging"
            nself.data.shape = (nelem, nself.data.shape[0] / nelem) + nself.data.shape[1:]
        return nself

    def splitLastDim(self, idxs, newdims):
        pass
        

    def mergeDim(self, matchpoint,newdim):
        matchpoint += 1
        assert len(self.idxs) > (matchpoint + 1), "Nested array not nested that deep"
        nself = self.copy()
        lidx = self.idxs[matchpoint]
        ridx = self.idxs[matchpoint+1]
        #Merging dims, 4 possibilities:
        #fixed-fixed, fixed-var, var-fixed, var-var
        if(isinstance(ridx,int)):
            nextpos,nextobj = nself._get_next_obj(matchpoint+1)
            if(isinstance(lidx,int)):  #fixed-fixed
                oshape = nextobj.shape
                nshape = oshape[:lidx] + (oshape[lidx] * oshape[ridx],) + oshape[(ridx+1):]
            else:  #var-fixed
                orshape = nextobj.shape[ridx]
                diff = (lidx[...,-1] - lidx[...,0]) * orshape
                nlidx = numpy.zeros(lidx.shape,dtype=lidx.dtype)
                nlidx[...,-1] = numpy.reshape(numpy.cumsum(diff.ravel()),diff.shape)
                nlidx[...,0] = nlidx[...,-1] - diff
                nself.idxs[matchpoint] = nlidx
                nshape =  (oshape[0] * oshape[ridx],) + oshape[(ridx+1):]
            nextobj = nextobj.reshape(nshape)
            nself._set_obj(nextpos,nextobj)
        else:
            if(isinstance(lidx,int)):  #fixed-var
                oshape = ridx.shape
                nshape = oshape[:-2] + (2 * oshape[-1],)
                nridx = ridx.reshape(nshape)[...,[0, -1]]
                nself.idxs[matchpoint] = nridx
            else:     #var-var
                nlidx = numpy.zeros(lidx.shape,dtype=lidx.dtype)
                nlidx[...,0] = ridx[lidx[...,0],0]
                nlidx[...,-1] = ridx[lidx[...,-1],-1]
                nself.idxs[matchpoint] = nlidx
        del nself.idxs[matchpoint+1]

        nself.dims = nself.dims[:(matchpoint-1)] + (newdim,) + nself.dims[(matchpoint+1):]
        return nself

    def swapDim(self, matchpoint, nldim, nrdim):
        matchpoint += 1
        assert len(self.idxs) > (matchpoint + 1), "Nested array not nested that deep"
        dep = self.dims[matchpoint+1].dependent 
        assert not dep or dep[-1] is False, "Switching dimensions that are dependent not possible"
        nself = self.copy()
        nself._swapDim(matchpoint)
        nself.dims[matchpoint] = nldim
        nself.dims[matchpoint+1] = nrdim
        return nself

    def _swapDim(self, matchpoint):#{{{
        nself = self
        lidx = self.idxs[matchpoint]
        ridx = self.idxs[matchpoint+1]
        #Merging dims, 4 possibilities:
        #fixed-fixed, fixed-var, var-fixed, var-var
        if(isinstance(lidx,int)):
            nextpos,nextobj = nself._get_next_obj(matchpoint+1)
            if(isinstance(ridx,int)):  #fixed-fixed
                pidx = range(len(nextobj.shape))
                tmp = pidx[lidx]
                pidx[lidx] = pidx[ridx]
                pidx[ridx] = tmp
                nextobj = numpy.transpose(nextobj,axes=pidx)
            else: #fixed-var --> var-fixed
                fixshape = ridx.shape[-2]
                nnextobj = numpy.zeros(nextobj.shape,dtype=nextobj.dtype)
                nnextobj = numpy.reshape(nnextobj, (nextobj.shape[0] / fixshape,fixshape) + nextobj.shape[1:])

                source_ridx = dimpaths.flatFirstDims(ridx,len(ridx.shape) -2).copy()
                
                ridx = ridx[...,0,:]
                oshape = ridx.shape
                diff = ridx[...,1] - ridx[...,0]
                ridx[...,1] = numpy.reshape(numpy.cumsum(diff.ravel()),diff.shape)
                ridx[...,0] = ridx[...,1] - diff

                dest_ridx = dimpaths.flatFirstDims(ridx,len(ridx.shape)-2)
                range_fixshape = range(fixshape)
                for i in range(0,len(dest_ridx)):
                    dest = slice(dest_ridx[i,0],dest_ridx[i,1])
                    for j in range_fixshape:
                        xj = j + i * fixshape
                        nnextobj[dest,j] = nextobj[source_ridx[xj,0]:source_ridx[xj,1]]
                nextobj = nnextobj

                nself.idxs[matchpoint] = ridx
                nself.idxs[matchpoint + 1] = 0 
                for i in range(matchpoint + 1, nextpos):
                    nself.idxs[i] = i -(matchpoint + 1)
            nself._set_obj(nextpos,nextobj)
            nself = nself._normalize(nextpos)
        else:
            nextpos,nextobj = nself._get_next_obj(matchpoint+1)
            if(isinstance(ridx,int)): #var-fixed --> fixed-var
                fixshape = nextobj.shape[1]
                
                nnextobj = numpy.zeros(nextobj.shape,dtype=nextobj.dtype)
                nnextobj = dimpaths.flatFirstDims(nnextobj,1)

                source_lidx = dimpaths.flatFirstDims(lidx,len(lidx.shape)-2)
                
                diff = source_lidx[:,1] - source_lidx[:,0]
                diff = numpy.repeat(diff,fixshape)
                nlidx = numpy.zeros((len(diff),2),dtype=int)
                nlidx[:,1] = numpy.cumsum(diff)
                nlidx[:,0] = nlidx[:,1] -diff
                nlidx.shape = lidx.shape[:-1] + (fixshape,2)

                dest_lidx = dimpaths.flatFirstDims(nlidx,len(nlidx.shape)-2)
                range_fixshape = range(fixshape)
                for i in range(0,len(source_lidx)):
                    src = slice(source_lidx[i,0],source_lidx[i,1])
                    for j in range_fixshape:
                        xj = j + i * fixshape
                        nnextobj[dest_lidx[xj,0]:dest_lidx[xj,1]] = nextobj[src,j]
                nextobj=nnextobj
                
                nself.idxs[matchpoint] = len(nlidx.shape) - 2
                nself.idxs[matchpoint+1] = nlidx

                for i in range(matchpoint +2, nextpos):
                    nself.idxs[i] = i - (matchpoint + 1)
            else: #var-var --> var-var
                nnextobj = numpy.zeros(nextobj.shape,dtype=nextobj.dtype)
                source_lidx = dimpaths.flatFirstDims(lidx,len(lidx.shape)-2)
                source_ridx = dimpaths.flatFirstDims(ridx,len(ridx.shape)-2)

                ldiff = source_lidx[:,1] - source_lidx[:,0]
                rdiff = source_ridx[source_lidx[:,0],1] - source_ridx[source_lidx[:,0],0]
                repeat = rdiff.copy()
                ldiff[repeat==0] = 0
                repeat[repeat==0] =1
                ldiff = numpy.repeat(ldiff,repeat)

                nlidx = numpy.zeros((len(ldiff),2),dtype=int)
                nlidx[:,1] = numpy.cumsum(ldiff)
                nlidx[:,0] = nlidx[:,1] -ldiff

                nridx = numpy.zeros((len(rdiff),2),dtype=int)
                nridx[:,1] = numpy.cumsum(rdiff)
                nridx[:,0] = nridx[:,1] -rdiff
                nridx.shape = lidx.shape
              
                pidx = range(len(nextobj.shape)+1)
                pidx[0] = 1
                pidx[1] = 0
                for i in range(0,len(source_lidx)):
                    lsrc = slice(source_lidx[i,0],source_lidx[i,1])
                    rsrc = slice(source_ridx[lsrc.start,0],source_ridx[lsrc.stop-1,1])
                    leftlen = lsrc.stop - lsrc.start
                    if(leftlen == 0):
                        rightlen= (rsrc.stop - rsrc.start)
                    else:
                        rightlen= (rsrc.stop - rsrc.start) / leftlen

                    d = numpy.reshape(nextobj[rsrc],(leftlen,rightlen) +  nextobj.shape[1:])
                    d = numpy.transpose(d,pidx)
                    nnextobj[rsrc] = d.ravel()
                nextobj = nnextobj

                nself.idxs[matchpoint] = nridx
                nself.idxs[matchpoint+1] = nlidx
            nself._set_obj(nextpos,nextobj)
            nself = nself._normalize(matchpoint + 2)#}}}

    def permuteDims(self, permute_idxs, ndims):
        nself = self.copy()
        assert len(permute_idxs) == (len(self.idxs) - 1), "Permute idxs do not cover all dimensions"
      
        #determine new dim positions
        dim_permute = [0] * len(permute_idxs)
        for pos, pi in enumerate(permute_idxs):
            dim_permute[pi] = pos
        permute_idxs = dim_permute

        for i in range(len(permute_idxs)):
            for j in range(len(permute_idxs) - i - 1):
                if permute_idxs[j] > permute_idxs[j+1]:
                    dep = self.dims[j+2].dependent 
                    nself._swapDim(j+1) #correct for first idx
                    permute_idxs[j],permute_idxs[j+1] = permute_idxs[j+1],permute_idxs[j]

        nself.dims = nself.dims[:1] + ndims
        return nself

    def _normalize(self, curpos):
        if(curpos >= len(self.idxs)):
            return self

        nself = self
        curpos,curobj = self._get_next_obj(curpos-1)
        nextpos,nextobj = self._get_next_obj(curpos)
        while curpos < len(self.idxs):
            respos = 0
            res = []
            oshape = curobj.shape 
            curobj = dimpaths.flatFirstDims(curobj, len(curobj.shape) - 2)
            nidx = numpy.zeros(curobj.shape,dtype=curobj.dtype)
            for rowpos in xrange(len(curobj)):
                start = curobj[rowpos,0]
                stop = curobj[rowpos,-1]
                res.append(nextobj[start:stop])
                nidx[rowpos,0] = respos
                respos += stop - start
                nidx[rowpos,1] = respos
            nextobj = numpy.concatenate(res)
            nidx.shape  = oshape
            nself._set_obj(curpos, nidx)
            nself._set_obj(nextpos, nextobj)

            curpos,curobj = self._get_next_obj(curpos)
            nextpos,nextobj = self._get_next_obj(nextpos)
        return nself

            
    def broadcast(self, repeat_dict, dim_dict):
        if not repeat_dict:
            return self
        nself = self.copy()

        repeats = [1] * len(self.idxs)
        ndims = list(nself.dims)
        for pos,repeat  in repeat_dict.iteritems():
            repeats[pos + 1] = repeat
            ndims[pos + 1] = dim_dict[pos]

        tilerepeats = []
        prev_repeat = False
        for pos,idx,repeat in zip(range(len(repeats)),nself.idxs,repeats):
            if(isinstance(idx,int)):
                if(isinstance(repeat,int)):
                    tilerepeats.append(repeat)
                else:
                    nextpos,nextobj = nself._get_next_obj(pos)
                    nself._apply_tile_broadcast(tilerepeats,nextpos,prev_repeat)
                    nextpos,nextobj = nself._get_next_obj(pos)
                    
                    #perform some checks
                    ntr = len(tilerepeats)
                    assert nextobj.shape[ntr] == 1, "Varbroadcast on full dimension not possible"
                    assert repeat.shape[:-1] == nextobj.shape[:ntr],"Index shapes should match"

                    #collapse first dims
                    nextobj = dimpaths.flatFirstDims(nextobj,ntr)

                    #replace new idx
                    nself.idxs[pos] = repeat
                  
                    #calculate var array lengths
                    temp_repeat = dimpaths.flatFirstDims(repeat,len(repeat.shape)-2)
                    varlength = temp_repeat[:,1] - temp_repeat[:,0]

                    #repeat nextobj
                    nextobj = numpy.repeat(nextobj,varlength,axis=0)

                    if(nextpos == len(self.idxs)):
                        nself.data = nextobj
                    else:
                        nself.idxs[pos] = nextobj
                    #update idxs 
                    for p in range(pos+1,nextpos):
                        nself.idxs[p] = p - pos

                    prev_repeat=True
                    tilerepeats = [1]
            else:
                assert repeat == 1, "Broadcast of full dimension not possible"
                nself._apply_tile_broadcast(tilerepeats,pos,prev_repeat)
                prev_repeat=True
                tilerepeats = [1]
        
        if(tilerepeats):
            nself._apply_tile_broadcast(tilerepeats,len(self.idxs),prev_repeat)
        nself.dims = dimpaths.DimPath(*ndims)
        return nself

    def _get_next_obj(self,pos):
        pos += 1
        while(len(self.idxs) > pos and isinstance(self.idxs[pos],int)):
            pos += 1
        if(pos >= len(self.idxs)):
            return (pos,self.data)
        else:
            return (pos,self.idxs[pos])

    def _set_obj(self,pos,obj):
        if(pos == len(self.idxs)):
            self.data = obj
        else:
            self.idxs[pos] = obj
        
            
    def _apply_tile_broadcast(self, tilerepeats, pos, prevrepeat):
        if(not tilerepeats and not prevrepeat):
            return
        if(pos == len(self.idxs)):
            idx = self.data
        else:
            idx = self.idxs[pos]
       
        assert isinstance(idx,numpy.ndarray), "Tilerepeats should be applied to numpy idx array"
        tilerepeats = list(tilerepeats)
        while(len(tilerepeats) < len(idx.shape)):
            tilerepeats.append(1)
        
        assert len(tilerepeats) == len(idx.shape), "Number of repeats does not match shape"

        shapeok = [col == 1 for tr,col in zip(tilerepeats,idx.shape) if tr > 1]
        if(not shapeok and not prevrepeat):#nothing to broadcast
            return
      
        assert all(shapeok), "Repeat of full dimension: error in  broadcast"

        if(pos == len(self.idxs)): #apply to data
            self.data = numpy.tile(self.data,tilerepeats)
            return
         
        idx = numpy.tile(idx,tilerepeats)

        lastrepeat = 0
        for tpos in xrange(len(tilerepeats)):
            if(tilerepeats[tpos] > 1):
                lastrepeat = tpos

        lastrepeat += 1
        oshape = idx.shape
        nshape = (numpy.multiply.reduce(oshape[:lastrepeat]),numpy.multiply.reduce(oshape[lastrepeat:]))
        idx.shape = nshape

        #make contigious (slices in order), affects nextobj
        nextpos, nextobj = self._get_next_obj(pos)
        curpos = 0
        res = []
        nidx = numpy.zeros(idx.shape,dtype=idx.dtype)
        for rowpos in xrange(len(idx)):
            start = idx[rowpos,0]
            stop = idx[rowpos,-1]
            res.append(nextobj[start:stop])
            nidx[rowpos,:] += curpos - start
            curpos += stop - start
        nextobj = numpy.concatenate(res)
        nidx.shape  = oshape
        if(nextpos == len(self.idxs)):
            self.data = nextobj
        else:
            self.idxs[nextpos] = nextobj
        self.idxs[pos] = nidx
        



    def __repr__(self):
        return "NestedArray < \n" + \
               "Idxs: " + str(self.idxs) + "  Dims: " + str(self.dims) + " Shape: " + str(self.data.shape) + "\n" + \
               "Data: " + str(self.data) + "  Dtype: " + str(self.data.dtype) + "\n" + \
               ">\n"



def co_mapseq(func, nested_arrays, *args, **kwargs):
    restype= kwargs.pop("res_type")
    dtype=restype.toNumpy()
    bc_allow=kwargs.pop("bc_allow",False)
    dimpath_set = set([na.dims[1:] for na in nested_arrays])
    data = []
    if not len(dimpath_set) == 1:
        if(not bc_allow is True):
            raise RuntimeError, "Nested arrays should have same dimensions!"
        
        udimpath = dimpaths.uniqueDimPath(dimpath_set)
        if(udimpath is False):
            raise RuntimeError, "Nested arrays should have same dimensions on same positions!"
        ndims = len(udimpath) + 1
        
        minlen = ndims
        for na in nested_arrays:
            if(len(na.dims) == ndims):
                na_ref = na
            minlen = min(len(na.dims),minlen)
        while(minlen < ndims and not isinstance(na_ref.getDimShape(minlen-1),int)):
            minlen += 1

        for pos,na in enumerate(nested_arrays):
            rep_dict = dict()
            dim_dict = dict()
            while(len(na.dims) < ndims):
                extpos = len(na.dims) - 1
                ndim = dimensions.Dim(1)
                na = na.insertDim(extpos,ndim)
                idx = na_ref.getDimShape(extpos)
                if(not isinstance(idx,int)):
                    rep_dict[extpos] = idx
                    dim_dict[extpos] = udimpath[extpos]
            na = na.broadcast(rep_dict,dim_dict)
            if(len(na.dims) > minlen):
                na = na.pack(rtypes.unknown,len(na.dims) - minlen)
            seq, flatshape = na._flatData()
            data.append(seq)
    else:
        na_ref = nested_arrays[0]
        for na in nested_arrays:
            seq,flatshape = na._flatData()
            data.append(seq)

    seq = func(data,*args, **kwargs)
    seq.shape = flatshape + seq.shape[1:]

    nself = na_ref.copy()
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

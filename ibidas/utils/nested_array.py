import numpy
import copy

_delay_import_(globals(),"..utils","cutils","sparse_arrays","util")
_delay_import_(globals(),"..utils.missing","Missing")
_delay_import_(globals(),"..itypes","dimpaths","rtypes","dimensions")
class NestedArray(object):
    def __init__(self,data,cur_type):
        self.data = util.darray([data],object)
        self.cur_type = cur_type
        self.idxs = [0]
        
    def copy(self):
        res = copy.copy(self)
        res.data = self.data.view(numpy.ndarray)
        res.idxs = list(self.idxs)
        return res
  
    def getDimShape(self,pos):
        pos += 1
        if(isinstance(self.idxs[pos],int)):
            nextpos,nextobj = self._get_next_obj(pos)
            return nextobj.shape[self.idxs[pos]]
        else:
            d = self.idxs[pos]
            d = d[...,1] - d[...,0]
            return d

    def _curIdxDepth(self):
        if(isinstance(self.idxs[-1],int)):
            return self.idxs[-1] + 1
        else:
            return 1

    def flat(self):
        data,rshape = self._flatData()
        data= util.darray([subelem for subelem in data],object,1,1)
        return data

    def _flatData(self,depth=-1):
        seq = self.data
        if(isinstance(self.idxs[depth],int)):
            rshape = seq.shape[:(self.idxs[depth]+1)]
            seq = dimpaths.flatFirstDims(seq,self.idxs[depth])
        else:
            #assert depth == -1 or depth==(len(self.idxs)-1),"Cannot flat data on non-fixed dim"
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
            
            if(len(dimpath) > cdepth):
                dtype = object
            elif(len(dimpath) < cdepth):
                dtype = dimpaths.getNestedArraySubType(subtype, cdepth - len(dimpath)).toNumpy()
            else:
                dtype = subtype.toNumpy()


            if(variable):
                res = []
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
                    if(not isinstance(elem,numpy.ndarray)):
                        elem = util.darray(list(elem),dtype,cdepth,cdepth)
                    else:
                        elem = validate_array(elem,cdepth,dtype)
                        assert len(elem.shape) == cdepth, "Number of dimensions incorrect"
                    res.append(elem)
                if not res:
                    ndata = util.darray([])
                else:
                    ndata = numpy.concatenate(res)
            else:
                if(cdepth == 1):
                    r = []
                    for elem in cdata:
                        r.extend(elem)
                    ndata = util.darray(r,dtype,1,1)
                else:
                    res = []
                    for elem in cdata:
                        #check that elem shape is not smaller or larger than expected 
                        if(not isinstance(elem,numpy.ndarray)):
                            elem = util.darray(list(elem),dtype,cdepth,cdepth)
                        else:
                            elem = validate_array(elem,cdepth,dtype)
                        res.append(elem)
                    if not res:
                        ndata = util.darray([])
                    else:
                        ndata = numpy.concatenate(res)


            if(variable):
                idxres.shape =  data.shape + (2,)
                nself.idxs.append(idxres)
            else:
                assert (len(cdata) == 0) or ((ndata.shape[0] % len(cdata)) == 0), "Leftover elements in joining dimensions"
                if len(cdata):
                    pshape = ndata.shape[0] / len(cdata)
                else:
                    pshape = 0
                ndata.shape = data.shape + (pshape,) + ndata.shape[1:]
                nself.idxs.append(len(data.shape))
            data = ndata    
            depth -= 1
            dimpath = dimpath[1:]
        
        nself.cur_type = subtype
        nself.data = data
        return nself
    
    def pack(self, subtype, depth=1):
        #init
        nself = self.copy()
      
        while(depth):
            assert len(nself.idxs) > 1, "Pack operation on nestedarray without index?!"
            idx = nself.idxs.pop()

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
                nself.data = util.darray(res,object)
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
                nseq = util.darray(res,dtype)
                return nseq
        else:
            def wrapfunc(seq, *args, **kwargs):
                res = [func(elem, *args, **kwargs) for elem in seq]
                nseq = util.darray(res,dtype)
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

    def insertDim(self,matchpoint):
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
        return nself

    def mergeAllDims(self):
        nself = self.copy()
        nself.idxs = [0,1]
        nself.data = self._flatData()[0]
        nself.data.shape = (1,) + nself.data.shape
        return nself
        
    def _diffToIdx(self,diff):
        nlidx = numpy.zeros(diff.shape + (2,),dtype=diff.dtype)
        nlidx[...,-1] = numpy.reshape(numpy.cumsum(diff.ravel()),diff.shape)
        nlidx[...,0] = nlidx[...,-1] - diff
        return nlidx


    def splitDim(self,matchpoint,lshape,rshape=None):
        
        #info on current dim
        cshape = self.getDimShape(matchpoint)
        matchpoint += 1
        cidx = self.idxs[matchpoint]
        
        nself = self.copy()
        nextpos,nextobj = nself._get_next_obj(matchpoint)

        if(rshape is None):
            if(isinstance(lshape,int)):
                if isinstance(cshape,int):
                    assert cshape % lshape == 0, "Cannot find matching right shape"
                    rshape = cshape / lshape
                else:
                    assert (cshape % lshape == 0).all(), "Cannot find matching right shape"
                    rshape = cshape / lshape
            else:
                raise RuntimeError, "Cannot determine right shape"

        if(isinstance(cidx,int)):
            if(isinstance(lshape,int)):
                if(isinstance(rshape,int)):
                    xshape = list(nextobj.shape)
                    assert lshape * rshape == xshape[cidx], "Splitted dimensions do not match size original dimension"
                    xshape[cidx] = rshape
                    xshape.insert(cidx,lshape)
                    nextobj = numpy.reshape(nextobj,xshape)
                    nself.idxs.insert(matchpoint,cidx)
                    for i in range(matchpoint+1,nextpos+1):
                        nself.idxs[i] += 1
                    nself._set_obj(nextpos+1,nextobj)
                else:
                    rshape = numpy.reshape(rshape,nextobj.shape[:cidx] + (lshape,))
                    assert (numpy.sum(rshape,axis=-1) == nextobj.shape[cidx]).all(), "Variable dimension lengths of rshape unequal to current shape"
                    rshape = self._diffToIdx(rshape)
                    nself.idxs[matchpoint] = rshape 
                    nself.idxs.insert(matchpoint,len(rshape.shape) - 2)
                    nextobj = dimpaths.flatFirstDims(nextobj,cidx)
                    for pos,i in enumerate(range(matchpoint+2,nextpos+1)):
                        nself.idxs[i] += pos
                    nself._set_obj(nextpos+1,nextobj)
            else:           
                #FIXME
                raise RuntimeError,"Splitting fixed to variable dimension not yet supported"
        else:
            if(isinstance(lshape,int)):
                if(isinstance(rshape,int)):
                    assert (lshape * rshape == nextobj.shape[0]), "Splitted dimensions do not match size original dimensions"
                    nextobj = nextobj.reshape(cidx.shape[:-1] + (lshape,rshape) + nextobj.shape[1:])
                    if(isinstance(nself.idxs[matchpoint-1],int)):
                        spos = nself.idxs[matchpoint-1] + 1
                    else:
                        spos = 0
                    nself.idxs[matchpoint] = spos+1
                    nself.idxs.insert(matchpoint,spos)
                    for i in range(matchpoint + 2, nextpos+1):
                        nself.idxs[i] += spos + 1
                    nself._set_obj(nextpos+1,nextobj)
                else:
                    assert nextobj.shape[0] == numpy.sum(rshape), "Splitted dimensions do not match size original dimension"
                    rshape = numpy.reshape(rshape,cidx.shape[:-1] + (lshape,))
                    rshape = self._diffToIdx(rshape)
                    nself.idxs[matchpoint] = rshape 
                    nself.idxs.insert(matchpoint,len(rshape.shape) - 2)
            else:
                if(isinstance(rshape,int)):
                    assert nextobj.shape[0] == numpy.sum(lshape) * rshape, "Splitted dimensions do not match size original dimension"
                    lshape = numpy.reshape(lshape,cidx.shape[:-1])
                    lshape = self._diffToIdx(lshape)
                    nextobj = nextobj.reshape((nextobj.shape[0] / rshape,rshape) + nextobj.shape[1:])
                    nself.idxs[matchpoint] = 0
                    nself.idxs.insert(matchpoint,lshape)
                    for i in range(matchpoint+2,nextpos+1):
                        nself.idxs[i] += 1
                    nself._set_obj(nextpos+1,nextobj)
                else:
                    rshape = rshape.ravel()
                    assert numpy.sum(lshape) == rshape.shape[0],"Splitted dimensions do not match size original dimension"
                    assert nextobj.shape[0] == numpy.sum(rshape), "Splitted dimensions do not match size original dimension"
                    lshape = numpy.reshape(lshape,cidx.shape[:-1])
                    lshape = self._diffToIdx(lshape)
                    rshape = self._diffToIdx(rshape)
                    nself.idxs[matchpoint] = rshape
                    nself.idxs.insert(matchpoint,lshape)
        return nself

    def mergeDim(self, matchpoint, result_fixed=True):
        matchpoint += 1
        assert len(self.idxs) > (matchpoint + 1), "Nested array not nested that deep"
        nself = self.copy()
        lidx = self.idxs[matchpoint]
        ridx = self.idxs[matchpoint+1]
        #Merging dims, 4 possibilities:
        #fixed-fixed, fixed-var, var-fixed, var-var
        if(isinstance(ridx,int)):
            nextpos,nextobj = nself._get_next_obj(matchpoint+1)
            oshape = nextobj.shape
            if(isinstance(lidx,int)):  #fixed-fixed
                nshape = oshape[:lidx] + (oshape[lidx] * oshape[ridx],) + oshape[(ridx+1):]
            else:  #var-fixed
                diff = (lidx[...,-1] - lidx[...,0]) * oshape[ridx]
                nlidx = numpy.zeros(lidx.shape,dtype=lidx.dtype)
                nlidx[...,-1] = numpy.reshape(numpy.cumsum(diff.ravel()),diff.shape)
                nlidx[...,0] = nlidx[...,-1] - diff
                nself.idxs[matchpoint] = nlidx
                nshape =  (oshape[0] * oshape[ridx],) + oshape[(ridx+1):]
            for i in range(matchpoint + 1, nextpos):
                nself.idxs[i] = nself.idxs[i] -1
            nextobj = nextobj.reshape(nshape)
            nself._set_obj(nextpos,nextobj)
        else:
            if(isinstance(lidx,int)):  #fixed-var
                oshape = ridx.shape
                nshape = oshape[:-2] + (2 * oshape[-2],)
                nridx = ridx.reshape(nshape)[...,[0, -1]]
                nself.idxs[matchpoint] = nridx
            else:     #var-var
                nlidx = numpy.zeros(lidx.shape,dtype=lidx.dtype)
                if(len(ridx) > 0):
                    nlidx[...,0] = ridx[lidx[...,0],0]
                    nlidx[...,-1] = numpy.roll(nlidx[...,0],-1)
                    nlidx.ravel()[-1] = ridx.ravel()[-1]
                nself.idxs[matchpoint] = nlidx
        del nself.idxs[matchpoint+1]
        if(not isinstance(nself.idxs[matchpoint],int) and result_fixed):
            nself._var_to_fixed(matchpoint), 
        return nself

    def _var_to_fixed(self, pos):
        idx = self.idxs[pos]
        nextpos,nextobj = self._get_next_obj(pos)
        diff = idx[...,-1] - idx[...,0]
        diffset = set(diff.ravel())

        assert len(diffset)==1, "Multiple lengths encountered while expecting fixed dim"
        
        nshape = diffset.pop()
        nextobj = nextobj.reshape(diff.shape + (nshape,) + nextobj.shape[1:])

        if(pos > 0 and isinstance(self.idxs[pos-1],int)):
            prevpos = self.idxs[pos-1]
        else:
            prevpos = 0
            
        for i in range(pos, nextpos):
            self.idxs[i] = (i - pos) + prevpos + 1
        self._set_obj(nextpos,nextobj)

    def mergeLastDims(self,depth):
        curdepth = len(self.idxs) - 1
        assert depth < curdepth, "Attempted merge to deep"
        
        nself = self
        nshapes = []
        for d in range(curdepth - depth, curdepth)[::-1]:
            s1 = nself.getDimShape(d-1)
            s2 = nself.getDimShape(d)
            nshapes.append((s1,s2))
            nself = nself.mergeDim(d-1, True)
        return nself, nshapes[::-1]

    def splitLastDim(self, shapes):
        nself = self
        for shapeleft, shaperight in shapes:
            nself = nself.splitDim(len(nself.idxs) - 2, shapeleft,shaperight)
        return nself

    def swapDim(self, matchpoint):
        matchpoint += 1
        assert len(self.idxs) > (matchpoint + 1), "Nested array not nested that deep"
        nself = self.copy()
        nself._swapDim(matchpoint)
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
                #added copy to ensure that array is c-contiguous,
                #as not being contiguous kills performance
                #In the future, maybe we should move contigious checks
                #to the operations that benefit from them, then we 
                #can remove this copy (and the copy below in var-var)
                nextobj = numpy.transpose(nextobj,axes=pidx).copy()
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
                for i in range(matchpoint + 1, nextpos):
                    nself.idxs[i] = i -(matchpoint + 1) + 1
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
                    d = numpy.transpose(d,pidx).copy()
                    nnextobj[rsrc] = d.ravel()
                nextobj = nnextobj

                nself.idxs[matchpoint] = nridx
                nself.idxs[matchpoint+1] = nlidx
            nself._set_obj(nextpos,nextobj)
            nself = nself._normalize(matchpoint + 2)#}}}

    def permuteDims(self, permute_idxs):
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
                    nself._swapDim(j+1) #correct for first idx
                    permute_idxs[j],permute_idxs[j+1] = permute_idxs[j+1],permute_idxs[j]

        return nself

           
    def broadcast(self, repeat_dict):
        if not repeat_dict:
            return self
        nself = self.copy()

        repeats = [1] * len(self.idxs)
        for pos,repeat  in repeat_dict.iteritems():
            repeats[pos + 1] = repeat
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
                    bcshape = list(nextobj.shape[:(ntr - len(repeat.shape))])
                    for l,d in zip(repeat.shape,nextobj.shape[(ntr-len(repeat.shape)):ntr]):
                        if(l != d):
                            assert l == 1, "Problem in broadcasting repeat shape"
                            bcshape.append(d)
                        else:
                            bcshape.append(1)
                    repeat = numpy.tile(repeat,bcshape) 

                    #collapse first dims
                    nextobj = dimpaths.flatFirstDims(nextobj,ntr)

                    #replace new idx
                    nself.idxs[pos] = self._diffToIdx(repeat)
                  
                    #repeat nextobj
                    nextobj = numpy.repeat(nextobj,repeat.ravel(),axis=0)
                    nself._set_obj(nextpos,nextobj)

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
        return nself

    def _get_next_obj(self,pos):
        pos += 1
        while(len(self.idxs) > pos and isinstance(self.idxs[pos],int)):
            pos += 1
        if(pos >= len(self.idxs)):
            return (pos,self.data.copy())
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

        shapeok = [col == 1 for tr,col in zip(tilerepeats,idx.shape) if tr != 1]
        if(not shapeok and not prevrepeat):#nothing to broadcast
            return
      
        assert all(shapeok), "Repeat of full dimension: error in  broadcast"

        if(pos == len(self.idxs)): #apply to data
            self.data = numpy.tile(self.data,tilerepeats)
            return
         
        idx = numpy.tile(idx,tilerepeats)

        oshape = idx.shape
        if(shapeok):
            lastrepeat = 0
            for tpos in xrange(len(tilerepeats)):
                if(tilerepeats[tpos] > 1):
                    lastrepeat = tpos

            lastrepeat += 1
            nshape = (numpy.multiply.reduce(oshape[:lastrepeat]),numpy.multiply.reduce(oshape[lastrepeat:]))
            idx.shape = nshape
        else:
            idx = dimpaths.flatFirstDims(idx,len(idx.shape) -2)
        #make contigious (slices in order), affects nextobj
        nextpos, nextobj = self._get_next_obj(pos)
        lastpos = 0
        res = []
        nidx = numpy.zeros(idx.shape,dtype=idx.dtype) + idx
        for rowpos in xrange(len(idx)):
            start = idx[rowpos,0]
            stop = idx[rowpos,-1]
            res.append(nextobj[start:stop])
            nidx[rowpos,:] += lastpos - start
            lastpos += stop - start
        if not res:
            nextobj = util.darray(res)
        else:
            nextobj = numpy.concatenate(res)
        nidx.shape  = oshape
        self._set_obj(nextpos,nextobj)
        self.idxs[pos] = nidx
        
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
            if not res:
                nextobj = util.darray(res)
            else:
                nextobj = numpy.concatenate(res)
            nidx.shape  = oshape
            nself._set_obj(curpos, nidx)
            nself._set_obj(nextpos, nextobj)

            curpos,curobj = self._get_next_obj(curpos)
            nextpos,nextobj = self._get_next_obj(nextpos)
        return nself

 


    def __repr__(self):
        return "NestedArray < \n" + \
               "Idxs: " + str(self.idxs) + " Shape: " + str(self.data.shape) + "\n" + \
               "Data: " + str(self.data) + "  Dtype: " + str(self.data.dtype) + "\n" + \
               ">\n"



def co_mapseq(func, nested_arrays, *args, **kwargs):
    restype= kwargs.pop("res_type")
    
    idxlen = set([len(na.idxs) for na in nested_arrays])
    assert len(idxlen) == 1, "Nested arrays should have same dimensions!"
    idxlen = idxlen.pop()

    na_ref = nested_arrays[0]

    #determine if there is incomplete broadcasting going on
    lastpos = len(na_ref.idxs)-1
    lasti = 0
    xshape = []
    for i in range(idxlen):
        pos = idxlen -i - 1

        nested = [not isinstance(na.idxs[pos],int) for na in nested_arrays]
        if all(nested):
            break
        dshapes = [na.getDimShape(pos-1) for na in nested_arrays]
        if any(nested):
            for dpos in xrange(len(dshapes)):
               if not isinstance(dshapes[dpos],int):
                  red = set(dshapes[dpos])
                  assert len(red) == 1, "Unequal dims (mixed nested/non-nested) in co_mapseq"
                  nested_arrays[dpos]._var_to_fixed(pos)
                  dshapes[dpos] = nested_arrays[dpos].getDimShape(pos-1)
                  assert isinstance(dshapes[dpos],int), "After conversion to fixed dim still nested dim"
          
        res = set(dshapes)
        if len(res) > 1:
            assert len(res) == 2 and 1 in res, "Unequal dims in co_mapseq"
            lastpos = pos - 1
            lasti = i
            res.discard(1)
            xshape.append(res.pop())
        else:
            xshape.append(res.pop())
    xshape = xshape[:(lasti+1)]
    bcdepth = len(na_ref.idxs) - lastpos

    dummy,flatshape = na_ref._flatData(depth=lastpos)
    data = []
    for na in nested_arrays:
        seq,dummy = na._flatData(depth=lastpos)
        if(bcdepth > 1 and len(seq.shape) > bcdepth): #if broadcasting, we want to explicitly pack data 
            nseq = dimpaths.flatFirstDims(seq,bcdepth-1)
            nseq = util.darray(list(nseq))
            nseq.shape = seq.shape[:bcdepth]
            seq = nseq
        data.append(seq)
    
    if(bcdepth > 1):
        kwargs["bcdepth"] = bcdepth
        flatshape = flatshape + tuple(xshape[::-1])

    seq = func(data,*args, **kwargs)
    if(isinstance(restype,tuple)):
        nselfs = []
        for elem,rtype in zip(seq,restype):
            elem.shape = flatshape + elem.shape[1:]
            nself = na_ref.copy()
            nself.data = elem
            nself.cur_type = rtype
            nselfs.append(nself)
        return nselfs
    else:
        seq.shape = flatshape + seq.shape[1:]
        nself = na_ref.copy()
        nself.data = seq
        nself.cur_type = restype
        return nself


def co_map(func, narrays, *args, **kwargs):
    restype= kwargs.get("res_type")
    if(isinstance(restype,tuple)):
        dtypes = [rtype.toNumpy() for rtype in restype]
        def wrapfunc(seqs, *args, **kwargs):
            if 'bcdepth' in kwargs:
                del kwargs['bcdepth']
                zipdata = numpy.broadcast(*seqs)
            else:
                zipdata = zip(*seqs)
            res = [func(elems, *args, **kwargs) for elems in zipdata]
            xres = []
            for pos,dtype in enumerate(dtypes):
                nseq = util.darray([row[pos] for row in res],dtype)
                xres.append(nseq)
            return tuple(xres)
    else: 
        dtype=restype.toNumpy()
        def wrapfunc(seqs, *args, **kwargs):
            if 'bcdepth' in kwargs:
                del kwargs['bcdepth']
                zipdata = numpy.broadcast(*seqs)
            else:
                zipdata = zip(*seqs)
            res = [func(elems, *args, **kwargs) for elems in zipdata]
            nseq = util.darray(res,dtype)
            return nseq
    return co_mapseq(wrapfunc, narrays, **kwargs)

def drop_prev_shapes_dim(ndata,shapes):
    cidx = ndata.idxs[-1]
    if(not isinstance(cidx,int)):
        cidx = 0
    sel = 0

    nshapes = []
    for lshape,rshape in shapes:
        if(isinstance(lshape,int)):
            if(not isinstance(rshape,int)):
                s = [slice(None,None)] * (cidx + 1)
                s[-1] = sel
                rshape = rshape[s]
                cidx = 0
                sel = slice(0,numpy.sum(rshape))
        else:
            s = [slice(None,None)] * (cidx + 1)
            s[-1] = sel
            lshape = lshape[s]
            cidx = 0
            sel = slice(0,numpy.sum(lshape))
            if(not isinstance(rshape,int)):
                s = [slice(None,None)] * (cidx + 1)
                s[-1] = sel
                rshape = rshape[s]
                cidx = 0
                sel = slice(0,numpy.sum(rshape))
        nshapes.append((lshape,rshape))
    return nshapes
 

def validate_array(seq, cdepth, dtype):
    if(len(seq.shape) < cdepth):
        oshape = seq.shape
        rem_ndims = cdepth - len(seq.shape) + 1
        if len(seq) == 0:
            seq = util.darray([],dtype)
            seq.shape = (0,) * rem_ndims
        else:
            seq = util.darray(list(seq.ravel()),dtype,rem_ndims,rem_ndims)
            seq.shape = oshape + seq.shape[1:]
        assert len(seq.shape) == cdepth, "Non array values encountered for dims " + str(dims[len(seq.shape):])
    elif(len(seq.shape) > cdepth):
        oshape = seq.shape
        seq = dimpaths.flatFirstDims(seq,cdepth-1)
        seq = util.darray([subelem for subelem in seq],object,1,1)
        seq.shape = oshape[:cdepth]

    if not seq.dtype == dtype:
        if(dtype.char == 'S' or dtype.char == 'U' or dtype.char == 'V'):
            z = numpy.zeros(seq.shape,dtype)
            z[:] = seq
            seq = z
        else:
            seq = numpy.cast[dtype](seq)
    return seq


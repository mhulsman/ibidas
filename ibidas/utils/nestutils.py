import numpy
from collections import defaultdict, Callable
from itertools import chain

_delay_import_(globals(),"cutils")
_delay_import_(globals(),"util")
_delay_import_(globals(),"missing","Missing")

def nestmap(data, inner_func, depth, outer_rep=numpy.dtype(object)):
    """Map function on nested data.

    :param data: data. Nested layers build using numpy object arrays.
    :param inner_func: applied on elements (Missing is filtered)
    :param depth: unnesting depth (0: func(data), 1: normal map, ..)
    :param outer_rep: numpy type used to package results of `inner_func`
               or function to package these results, res = f(seq)
    """
    inner_func = util.filter_missing(inner_func)
    
    if(isinstance(outer_rep, Callable)):
        outer_func = util.filter_missing(outer_rep)
    else:
        if(outer_rep  == object):
            outer_func = cutils.darray
        else:
            outer_func = lambda x: cutils.darray(x, outer_rep)

    return _nestmap(data, inner_func, depth, outer_func)

def _nestmap(data, inner_func, depth, outer_func):
    if(data is Missing):
        return Missing

    if(not depth):
        return outer_func([inner_func(data)])[0]
    elif(depth == 1):
        return outer_func([inner_func(elem) 
                   for elem in data]).view(util.farray)
    elif(depth == 2):
        return cutils.darray([outer_func([inner_func(subelem) 
                              for subelem in elem]) for elem in data]).view(util.farray)
    else:
        return cutils.darray([_nestmap(elem, inner_func, 
            depth - 1, outer_func) for elem in data]).view(util.farray)


def nest_broadcast(bcast, data, ddims, func, rtypes, rdims, pack=1, func_attr={}, vectorize=0):
    if(any([elem is Missing for elem in data])):
        return (Missing,) * len(rtypes)
    
    if(not bcast):
        return func(data, bcast, ddims, pack != len(rdims), rtypes, rdims, func_attr)

    pack_types = (object,) * len(rtypes)

    nrdims = rdims[1:]
    if(nrdims and nrdims[0].variable):
        npack = 1
    else:
        if(not pack == 0):
            while pack < len(rdims) and not rdims[pack].variable:
                pack += 1
            if(pack == len(rdims)):
                pack_types = [type.toNumpy() for type in rtypes]
        npack = 0
   
    nbcast = bcast[1:]
    nddims = list(ddims)
    for pos in xrange(len(data)):
        if(bcast[0] & (2**pos)):
            nddims[pos] = nddims[pos][1:]

    if(len(data) == 1):
        if(vectorize and isinstance(data[0], numpy.ndarray) and len(bcast) <= len(data[0].shape)):
            res = func(data, bcast, ddims, pack != len(rdims), rtypes, rdims, func_attr)
        else:
            res = [nest_broadcast(nbcast, (elem,), nddims, func, rtypes, nrdims, npack, func_attr, vectorize) for elem in data[0]]
    elif(len(data) == 2):
        if(vectorize and len(bcast) < len(data[0].shape) and all([elem == 1 for elem in bcast])):
            res = func(data, bcast, ddims, pack != len(rdims), rtypes, rdims, func_attr)
        elif(((not bcast[0] & 1) or len(bcast) <= len(data[0].shape)) and \
             ((not bcast[0] & 2) or len(bcast) <= len(data[1].shape)) and len(set(bcast)) == 1
             and len(rtypes) == 1):
            nrdims = nrdims[len(nbcast):]
            pack = len(bcast)
            while pack < len(nrdims) and not nrdims[pack].variable:
                pack += 1
            if(pack == len(rdims)):
                pack_types = [type.toNumpy() for type in rtypes]
 
            data = list(data)
            if(bcast[0] == 1):
                rshape0 = data[0].shape[:len(bcast)]
                rem_shape0 = data[0].shape[len(bcast):]
                nddims[0] = nddims[0][len(nbcast):]
                data[0] = data[0].reshape((numpy.prod(rshape0),) + rem_shape0)
                res = [func((elem, data[1]), (), nddims, pack != len(nrdims), rtypes, nrdims, func_attr) for elem in data[0]]
                res = cutils.darray(res, pack_types[0])
                res.shape = rshape0
            elif(bcast[0] == 2):
                rshape1 = data[1].shape[:len(bcast)]
                rem_shape1 = data[1].shape[len(bcast):]
                nddims[1] = nddims[1][len(nbcast):]
                data[1] = data[1].reshape((numpy.prod(rshape1),) + rem_shape1)
                res = [func((data[0], elem), (), nddims, pack != len(nrdims), rtypes, nrdims, func_attr) for elem in data[1]]
                res = cutils.darray(res, pack_types[0])
                res.shape = rshape1
            else:
                rshape0 = data[0].shape[:len(bcast)]
                rem_shape0 = data[0].shape[len(bcast):]
                rshape1 = data[1].shape[:len(bcast)]
                rem_shape1 = data[1].shape[len(bcast):]
                nddims[0] = nddims[0][len(nbcast):]
                nddims[1] = nddims[1][len(nbcast):]
                data[0] = data[0].reshape((numpy.prod(rshape0),) + rem_shape0)
                data[1] = data[1].reshape((numpy.prod(rshape1),) + rem_shape1)
                res = [func(elems, (), nddims, pack != len(nrdims), rtypes, nrdims, func_attr) for elems in zip(*data)]
                res = cutils.darray(res, pack_types[0])
                res.shape = rshape0
        else:
            if(bcast[0] == 1):
                res = [nest_broadcast(nbcast, (elem, data[1]), nddims, func, rtypes, nrdims, npack, func_attr, vectorize) for elem in data[0]]
            elif(bcast[0] == 2):
                res = [nest_broadcast(nbcast, (data[0], elem), nddims, func, rtypes, nrdims, npack, func_attr, vectorize) for elem in data[1]]
            else:
                res = [nest_broadcast(nbcast, elems, nddims, func, rtypes, nrdims, npack, func_attr, vectorize) for elems in zip(*data)]
    else:
        res = []
        ndata = list(data)
        iterpos = [pos for pos, d in enumerate(data) if bcast[0] & (2**pos)]
        nestiters = [None] * len(data)
        for ipos in iterpos:
            nestiters[ipos] = iter(data[ipos])
        try:
            while True:
                for ipos in iterpos:
                    ndata[ipos] = nestiters[ipos].next()
                res.append(nest_broadcast(nbcast, ndata, nddims, func, rtypes, nrdims, npack, func_attr, vectorize))
        except StopIteration, e:
            pass
    
    if(len(rtypes) == 1):
        if(pack > 0):
            res = cutils.darray(res, pack_types[0], pack).view(util.farray)
    else:
        nres = []
        for col in range(len(rtypes)):
            if(pack > 0):
                coldata = cutils.darray([elem[col] for elem in res], pack_types[col], pack).view(util.farray)
            else:
                coldata = [elem[col] for elem in res]
                
            nres.append(coldata)
        res = tuple(nres)
    return res



def nestop(data, func, depths):
    """Binary map function on nested data.

    Performs function on simultaneously unpacked data

    :param data: sequence of (nested) data objects
    :param func: func to be applied on unpacked data 
            (each source as seperate parameter)
    :param depth: unnesting depth for each source. If unequal, 
           function from lower depths is broadcasted upward,
           i.e. elements are 'replicated'
    """
    if(Missing in data):
        return Missing
    if(max(depths) > 1):
        iterpos = [pos for pos, depth in enumerate(depths) if depth > 0]
        if(not iterpos):
            return func(*data)

        nestiters = [None] * len(depths)
        for ipos in iterpos:
            nestiters[ipos] = iter(data[ipos])
    
        ndata = list(data)
        nres = numpy.empty((len(data[ipos]),), object).view(util.farray)
        pos = 0
        try:
            ndepths = [depth -1 for depth in depths]
            while True:
                for ipos in iterpos:
                    ndata[ipos] = nestiters[ipos].next()
                nres[pos] = nestop(ndata, func, ndepths)
                pos += 1
        except StopIteration, e:
            return nres
    else:
        return func(*data)


def nest_filter(data, filter_data, filter_depth, fdims, cdims, slice_type):
    """Nested filter function

    Performs filtering on data using filter specified 
    by `filter_data`.

    :param data: (nested) data object
    :param filter_data: (nested) data object, of which inner objects 
                    are useable as index in numpy containers
    filter_depth: nest depth at which to apply filter_data
    fdims: new dimensions of `data` after filtering
    cdims: dimensions of `filter_data`
    slice_type: numpy dtype used to pack inner elements in `data`

    Note: fdims and cdims should be sequences of Dim objects
    they are used to determine when `filter_data` should be unpacked
    """

    if(data is Missing or filter_data is Missing):
        return Missing
    
    if(filter_depth == 0):
        if(isinstance(filter_data, numpy.ndarray) and filter_data.dtype == object):
            return data[numpy.cast[bool](filter_data)]
        else:
            return data[filter_data]

    nfdims = fdims[1:]
    if(not nfdims):
        res_type = slice_type
    else:
        res_type = object

    if(cdims and cdims[0] == fdims[0]):
        ncdims = cdims[1:]
        return cutils.darray([nest_filter(ndata, nfilter_data, 
                             filter_depth - 1, nfdims, ncdims, slice_type) 
                             for ndata, nfilter_data in zip(data, filter_data)],
                             res_type).view(util.farray)
    else:
        return cutils.darray([nest_filter(ndata, filter_data, 
                                filter_depth - 1, nfdims, cdims, slice_type) 
                        for ndata in data], res_type).view(util.farray)
def nest_crossjoin(join_broadcast, ldatacols, rdatacols, group):
    if(join_broadcast and (not group or len(join_broadcast) > 1)):
        njoin_broadcast = join_broadcast[1:]
        ndatacols = []
        lpos = [pos for pos, elem in enumerate(ldatacols) if not elem is Missing]
        rpos = [len(ldatacols) + pos for pos, elem in enumerate(rdatacols) if not elem is Missing]

        lmdatacols = [elem for elem in ldatacols if not elem is Missing]
        rmdatacols = [elem for elem in rdatacols if not elem is Missing]

        if(join_broadcast[0] == 1):
             for nldata in zip(*lmdatacols):
                 res = nest_crossjoin(njoin_broadcast, nldata, rdatacols, group)
                 ndatacols.append(res)
        elif(join_broadcast[0] == 3):
             for nldata, nrdata in zip(zip(*lmdatacols), zip(*rmdatacols)):
                 res = nest_crossjoin(njoin_broadcast, nldata, nrdata, group)
                 ndatacols.append(res)
        else:
            raise RuntimeError, "Unexepcted value in join_broadcast"

        res = [Missing] * (len(ldatacols) + len(rdatacols))
        for i, pos in enumerate(chain(lpos, rpos)):
            res[pos] = cutils.darray([elem[i] 
                         for elem in ndatacols]).view(util.farray)

    else:
        
        res = []
        if(group):
            res.extend(ldatacols)
            if(not join_broadcast):
                for fdata in rdatacols:
                    res.append(fdata)
            elif(join_broadcast[0] == 1):
                leftlen = len(ldatacols[0])
                for fdata in rdatacols:
                    res.append(cutils.darray([fdata] * leftlen).view(util.farray))
            elif(join_broadcast[0] == 3): #they are already a group
                 res.extend(rdatacols) 
            else: 
                raise RuntimeError, "Unexepcted value in join_broadcast"
        else:
            leftlen = len(ldatacols[0])
            rightlen = len(rdatacols[0])
            for fdata in ldatacols:
                res.append(numpy.repeat(fdata, rightlen).view(util.farray))
            for fdata in rdatacols:
                res.append(numpy.tile(fdata, leftlen))
    return res
       

def nest_equijoin(join_broadcast, ldatacols, rdatacols, lequicol, requicol, ldepth, rdepth, group):
    if(lequicol is Missing or requicol is Missing):
        return [Missing] * (len(ldatacols) + len(rdatacols))
   
    if(join_broadcast and (not group or len(join_broadcast) > 1 or join_broadcast[0] == 1)):
        njoin_broadcast = join_broadcast[1:]
        ndatacols = []
        lpos = [pos for pos, elem in enumerate(ldatacols) if not elem is Missing]
        rpos = [len(ldatacols) + pos for pos, elem in enumerate(rdatacols) if not elem is Missing]

        lmdatacols = [elem for elem in ldatacols if not elem is Missing]
        rmdatacols = [elem for elem in rdatacols if not elem is Missing]

        if(join_broadcast[0] == 1):
             if not 3 in njoin_broadcast:
                 nrequicol = defaultdict(list)
                 for pos, val in enumerate(requicol):
                     nrequicol[val].append(pos)
                 requicol = nrequicol
             if(ldepth > 0):
                for nldata, nlequicol in util.zip_broadcast(zip(*lmdatacols), lequicol):
                    res = nest_equijoin(njoin_broadcast, nldata, rdatacols, nlequicol, requicol, ldepth - 1, rdepth, group)
                    ndatacols.append(res)
             else:
                for nldata in zip(*lmdatacols):
                    res = nest_equijoin(njoin_broadcast, nldata, rdatacols, lequicol, requicol, ldepth, rdepth, group)
                    ndatacols.append(res)
        elif(join_broadcast[0] == 3):
             if(ldepth > 0 and rdepth > 0):
                for nldata, nrdata, nlequicol, nrequicol in util.zip_broadcast(zip(*lmdatacols), zip(*rmdatacols), lequicol, requicol):
                    res = nest_equijoin(njoin_broadcast, nldata, nrdata, nlequicol, nrequicol, ldepth - 1, rdepth - 1, group)
                    ndatacols.append(res)
             elif(ldepth > 0):
                for nldata, nrdata, nlequicol in util.zip_broadcast(zip(*lmdatacols), zip(*rmdatacols), lequicol):
                    res = nest_equijoin(njoin_broadcast, nldata, nrdata, nlequicol, requicol, ldepth - 1, rdepth, group)
                    ndatacols.append(res)
             elif(rdepth > 0):
                for nldata, nrdata, nrequicol in util.zip_broadcast(zip(*lmdatacols), zip(*rmdatacols), requicol):
                    res = nest_equijoin(njoin_broadcast, nldata, nrdata, lequicol, nrequicol, ldepth, rdepth - 1, group)
                    ndatacols.append(res)
             else:
                for nldata, nrdata in zip(zip(*lmdatacols), zip(*rmdatacols)):
                    res = nest_equijoin(njoin_broadcast, nldata, nrdata, lequicol, requicol, ldepth, rdepth, group)
                    ndatacols.append(res)
        else:
            raise RuntimeError, "Unexepcted value in join_broadcast"

        res = [Missing] * (len(ldatacols) + len(rdatacols))
        for i, pos in enumerate(chain(lpos, rpos)):
            res[pos] = cutils.darray([elem[i] 
                         for elem in ndatacols]).view(util.farray)

    else:
        if(not isinstance(requicol, dict)):
            nrequicol = defaultdict(list)
            for pos, val in enumerate(requicol):
                if not val is Missing:
                    nrequicol[val].append(pos)
            requicol = nrequicol
        res = []
        if(group):
            res.extend(ldatacols)
            if(not join_broadcast):
                pos = requicol.get(lequicol, [])
                for fdata in rdatacols:
                    res.append(fdata[pos])
            elif(join_broadcast[0] == 1):
                leftlen = len(ldatacols[0])
                if(ldepth > 0):
                    for fdata in rdatacols:
                        colres = []
                        for elem in lequicol:
                            pos = requicol.get(elem, [])
                            colres.append(fdata[pos])
                        res.append(cutils.darray(colres).view(util.farray))
                else:
                    pos = requicol.get(lequicol,[])
                    for fdata in rdatacols:
                        res.append(cutils.darray([fdata] * leftlen).view(util.farray))
            else: 
                raise RuntimeError, "Unexepcted value in join_broadcast"
        else:
            if(ldepth == 0):
                leftlen = len(ldatacols[0])
                pos = requicol.get(lequicol, [])
                rightlen = len(pos)
                for fdata in ldatacols:
                    res.append(numpy.repeat(fdata, rightlen).view(util.farray))
                for fdata in rdatacols:
                    res.append(numpy.tile(fdata[pos], leftlen))
            else:
                leftpos = []
                rightpos = []
                for lpos, elem in enumerate(lequicol):
                    rpos = requicol.get(elem, [])
                    rightpos.extend(rpos)
                    leftpos.extend([lpos] * len(rpos))
                for fdata in ldatacols:
                    res.append(fdata[leftpos])
                for fdata in rdatacols:
                    res.append(fdata[rightpos])
    return res

def nest_flatten(sdatas, depths, rtypes):
    if(max(depths) > 2):
        rdatas = [[] for elem in depths]
        ndepths = [depth -1 for depth in depths]

        for ndata in zip(*sdatas):
            res = nest_flatten(ndata, ndepths, rtypes)
            for elem, rdata in zip(res, rdatas):
                rdatas.append(elem)
        return [cutils.darray(elem).view(util.farray) 
                for elem in rdatas]     
    else:
        lposidx = depths.index(2)
        rdatas = [[] for elem in depths]
        for rowdatas in zip(*sdatas):
            length = len(rowdatas[lposidx])
            for elem, rdata, depth in zip(rowdatas, rdatas, depths):
                if(depth == 1):
                    rdata.extend([elem] * length)
                else:
                    rdata.extend(elem)
        return [cutils.darray(elem, rtype).view(util.farray) 
                for elem, rtype in zip(rdatas, rtypes)]     
                


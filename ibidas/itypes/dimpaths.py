import numpy

_delay_import_(globals(),"dimensions","Dim")

class DimPath(tuple):
    def __new__(cls, *dims):
        assert all([isinstance(dim, Dim) for dim in dims]),"DimPath should consist of dimensions"
        return tuple.__new__(cls, dims)
    
    def _getNames(self):
        return [dim.name for dim in self]
    names = property(fget=_getNames)

    def getDimIndexByName(self,name):
        dimnames = self.names
        try:
            return dimnames.index(name)
        except ValueError:
            return None

    def __getslice__(self,*idx):
        res = tuple.__getslice__(self,*idx)
        if(isinstance(res,tuple)):
            return DimPath(*res)
        return res

    def __getitem__(self,*idx):
        res = tuple.__getitem__(self,*idx)
        if(isinstance(res,tuple)):
            return DimPath(*res)
        return res

    def __repr__(self):
        return "DimPath" + tuple.__repr__(self)
    

    def __add__(self, other):
        return DimPath(*tuple.__add__(self,other))

    def __contains__(self,part):
        if(isinstance(part,DimPath)):
            pos = -1
            try:
                for dim in dim_path:
                    pos = sdims.index(dim, pos + 1)
            except ValueError:
                return False
            return True
        else:
            return tuple.__contains__(self,part)

    def changeNames(self,newname_dict):
        newpath = []
        for dim in self:
            if(dim in newname_dict):
                dim = dim.copy()
                dim.name = newname_dict[dim]
            elif(dim.name in newname_dict):
                dim = dim.copy()
                dim.name = newname_dict[dim.name]
            newpath.append(dim)
        return DimPath(newpath)

    def contigiousFixedNDims(self):
        depth = 0
        #maximize array size with fixed dims
        while(depth < len(self) and not self[depth].variable and not 
                self[depth].has_missing):
            depth+=1
        return depth
    
def commonDimPath(dimpaths):#{{{
    """Returns common dimensions shared by all slices"""
    pos = 0
    minlen = min([len(dp) for dp in dimpaths])
    while(pos < minlen):
        cdim = set([dp[pos] for slice in dimpaths])
        pos += 1
        if(len(cdim) != 1):
            break
    return dimpaths[0][:pos]#}}}

def flatFirstDims(array,ndim):
    oshape = array.shape
    ndim = ndim + 1
    rshape = (int(numpy.multiply.reduce(array.shape[:ndim])),) + array.shape[ndim:]
    return array.reshape(rshape)


import os
import csv
from logging import warning

import wrapper
import tsv
from ..itypes import rtypes
from .. import ops
from .. import repops_slice
from ..constants import *

import python
from ..itypes import detector, dimensions,rtypes, dimpaths
from ..utils import nested_array, util


class MatrixTSVRepresentor(tsv.TSVRepresentor):
    def __init__(self, filename, dialect=False, skiprows=-1, col_sections=[], row_sections=[]):
        file = util.open_file(filename)
  
        reader, dialect = self.getReader(file, dialect)
        
        #determine number of rows to skip (e.g. comments)
        if(skiprows == -1):
            skiprows = self.determineSkipRows(file, reader)

        for i in xrange(skiprows):
            file.readline()
        startpos = file.tell()

        #determine type
        colname, colsize = self.convertSection(col_sections, "col_")
        rowname, rowsize = self.convertSection(row_sections, "row_")
                    
        dim1 = dimensions.Dim(UNDEFINED, name="rows")
        dim2 = dimensions.Dim(UNDEFINED, name="cols")
        
        dim_string = dimensions.Dim(UNDEFINED)
        bytes = rtypes.TypeBytes(dims=dimpaths.DimPath(dim_string))

        types = []
        names = []
        for coln,cols in zip(colname, colsize):
            if cols == 1:
                t =rtypes.TypeArray(dims=dimpaths.DimPath(dim1),subtypes=(bytes,))
            else:
                dim = dimensions.Dim(cols, name=coln+"_dim")
                subtype = rtypes.TypeArray(dims=dimpaths.DimPath(dim),subtypes=(bytes,))            
                t = rtypes.TypeArray(dims=dimpaths.DimPath(dim1),subtypes=(subtype,))
            types.append(t)
            names.append(coln)
        for rown,rows in zip(rowname, rowsize):
            if rows == 1:
                t =rtypes.TypeArray(dims=dimpaths.DimPath(dim2),subtypes=(bytes,))
            else:
                dim = dimensions.Dim(rows, name=rown+"_dim")
                subtype = rtypes.TypeArray(dims=dimpaths.DimPath(dim2),subtypes=(bytes,))            
                t = rtypes.TypeArray(dims=dimpaths.DimPath(dim),subtypes=(subtype,))
            types.append(t)
            names.append(rown)
        subtype = rtypes.TypeArray(dims=dimpaths.DimPath(dim2),subtypes=(bytes,))            
        types.append(rtypes.TypeArray(dims=dimpaths.DimPath(dim1),subtypes=(subtype,)))
        names.append('matrix')

        dtype = rtypes.TypeTuple(subtypes=tuple(types), fieldnames=tuple(names))
        slice = MatrixTSVOp(filename, dialect, startpos, dtype, colsize, rowsize, "data")
        nslices = repops_slice.UnpackTuple._apply(slice)
        file.close()
        self._initialize(tuple(nslices))
    
    def convertSection(self, section, prefix):
        colsize = []
        colname = []
        for pos,c in enumerate(section):
            if isinstance(c,basestring):
                colsize.append(1)
                colname.append(c)
            elif isinstance(c,dict):
                assert len(c) == 1, "Only one element in col_section dicts allowed"
                colname.append(c.keys()[0])
                colsize.append(c.values()[0])
            elif isinstance(c, int):
                assert c >= 1, "Only matrix section sizes >= 1 allowed"
                colsize.append(c)
                if pos < 26:
                    colname.append(prefix + chr(pos + 97))
                else:
                    colname.append(prefix + str(pos))
        colname = [util.valid_name(coln) for coln in colname]

        return (colname, colsize)

class MatrixTSVOp(ops.ExtendOp):
    __slots__ = ["filename", "dialect","startpos","colsize","rowsize"]

    def __init__(self, filename, dialect, startpos, rtype, colsize, rowsize, name):
        self.filename = filename
        self.dialect = dialect
        self.startpos = startpos
        self.rowsize = rowsize
        self.colsize = colsize
        ops.ExtendOp.__init__(self,name=name,rtype=rtype)

    def py_exec(self):
        file = util.open_file(self.filename)
        file.seek(self.startpos)

        if(issubclass(self.dialect, csv.Dialect)):
            reader = csv.reader(file, self.dialect)
        else:
            reader = csv.reader(file, delimiter=self.dialect)
        data = [tuple(row) for row in reader]
        
        xlen = len(data[0])
        for pos, row in enumerate(data):
            if len(row) != xlen:
                warning('Removing last %d rows', len(data) - pos)            
                data = data[:pos]


        file.close()
        
        cols = []
        rows = []
        skiprows = sum(self.rowsize)
        for cs in self.colsize:
            ndata = []
            ncol = []
            if cs == 1:
                for row in data:
                    ndata.append(row[1:])
                    ncol.append(row[0])
            else:
                for row in data:
                    ndata.append(row[cs:])
                    ncol.append(row[:cs])
            cols.append(ncol[skiprows:])
            data = ndata
        
        for rs in self.rowsize:
            if rs == 1:
                nrow = data[0]
                data = data[1:]
            else:
                nrow = data[:rs]
                data = data[rs:]
            rows.append(nrow)
        
        data = tuple(cols) + tuple(rows) + (data,)
        ndata = nested_array.NestedArray(data,self.type)
        return python.ResultOp.from_slice(ndata,self)


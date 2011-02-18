import os
import csv

import wrapper
from ..itypes import rtypes
from .. import ops
from ..constants import *

_delay_import_(globals(),"wrapper_py")
_delay_import_(globals(),"..itypes","detector")
_delay_import_(globals(),"..utils","nested_array","util")

class TSVRepresentor(wrapper.SourceRepresentor):
    def __init__(self, filename, dialect=False, skiprows=-1, dtype=rtypes.unknown, fieldnames=None):
        file = util.open_file(filename)
   
        #determine dialect, create csv parser
        if(dialect is False):
            lines = []
            #get sample
            for line in file:
                lines.append(line)
                if(len(lines) > 500):
                    break
            #sniff last 20 lines
            lines = lines[-20:]
            dialect = csv.Sniffer().sniff("\n".join(lines))
            file.seek(0)
            reader = csv.reader(file, dialect)
        else:
            if(issubclass(dialect, csv.Dialect)):
                reader = csv.reader(file, dialect)
            else:
                reader = csv.reader(file, delimiter=dialect)
        
        #determine number of rows to skip (e.g. comments)
        if(skiprows == -1):
            splitsize = []
            for row in reader:
                splitsize.append(len(row))
                if(len(row[0]) == 0 or ((len(splitsize) == 1 or splitsize[-2] == 0) and (row[0][0] == '#' or row[0][0] == '%' or row[0][0] == "!"))):
                    splitsize[-1] = 0
                elif(len(splitsize) > 25):
                    x = set(splitsize[-10:])
                    if(len(x) == 1):
                        break
                elif(len(splitsize) > 500):
                    raise RuntimeError, "Cannot find correct number of columns. Incorrect delimiter?"                    
            
            real_split  = splitsize[-1]
            skiprows = 0
            for pos, split in enumerate(splitsize):
                if(split != real_split):
                    skiprows = pos + 1
            file.seek(0)

        for i in xrange(skiprows):
            file.readline()
        startpos = file.tell()

        #determine type
        if(dtype is None):
            dtype = rtypes.unknown
        elif(dtype == rtypes.unknown):
            if(fieldnames is None):
                sample = []
                for line in file:
                    sample.append(line)
                    if(len(sample) > 10):
                        break
                file.seek(startpos)
                fieldnames = reader.next()
                if(not csv.Sniffer().has_header("\n".join(sample))):
                    file.seek(startpos)
                    fieldnames = None
            elif(fieldnames == "auto"):
                fieldnames = reader.next()

            startpos = file.tell()
            #parse data
            data = [tuple(row) for row in reader]
            file.seek(startpos)

            det = detector.Detector()
            det.process(data)
            dtype = det.getType()
            if(not fieldnames is None and dtype != rtypes.unknown):
                assert isinstance(dtype,rtypes.TypeArray),"Error while determining type"
                assert isinstance(dtype.subtypes[0],rtypes.TypeTuple),"Error while determining type"
                dtype.subtypes[0].fieldnames = tuple(fieldnames)
        else:
            if(isinstance(dtype,basestring)):
                dtype = rtypes.createType(dtype)

        slice = TSVOp(filename, dialect, startpos, dtype, "data")
        if(slice.type.__class__ is rtypes.TypeArray):
            slice = ops.UnpackArrayOp(slice)
        if(slice.type.__class__ is rtypes.TypeTuple):
            nslices = [ops.UnpackTupleOp(slice, idx) for idx in range(len(slice.type.subtypes))]
        else:
            nslices = [slice]
 
        if(any([slice.type == rtypes.unknown for slice in nslices])):
            state = RS_SLICES_KNOWN
        else:
            state = RS_ALL_KNOWN
        file.close()
        self._initialize(tuple(nslices),state)


class TSVOp(ops.ExtendOp):
    __slots__ = ["filename", "dialect","startpos"]

    def __init__(self, filename, dialect, startpos, rtype, name):
        self.filename = filename
        self.dialect = dialect
        self.startpos = startpos
        ops.ExtendOp.__init__(self,name=name,rtype=rtype)

    def py_exec(self):
        file = util.open_file(self.filename)
        file.seek(self.startpos)

        if(issubclass(self.dialect, csv.Dialect)):
            reader = csv.reader(file, self.dialect)
        else:
            reader = csv.reader(file, delimiter=self.dialect)
        data = [tuple(row) for row in reader]
        file.close()
        ndata = nested_array.NestedArray(data,self.type)
        return wrapper_py.ResultOp.from_slice(ndata,self)


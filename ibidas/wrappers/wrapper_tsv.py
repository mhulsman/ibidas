import os
import csv


import wrappr
from ..itypes import rtypes

_delay_import_(globals(),"wrapper_py","Result","rep")
_delay_import_(globals(),"..utils","util")

def read_tsv(filename, dialect=False, skiprows=-1, type=rtypes.unknown, fieldnames=None):
    return TSVRepresentor(filename, dialect, skiprows, type, fieldnames)
    


class TSVRepresentor(wrapper.SourceRepresentor):
    _select_indicator = None
    _select_executor = None
    
    def __init__(self, filename, dialect, skiprows, type, fieldnames):
        file = open(filename)

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
            if(isinstance(dialect, csv.Dialect)):
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

        #determine fieldnames
        if(not fieldnames):
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

        #parse data
        data = [tuple(row) for row in reader]
        self._res = rep(data, type)
        
        #assign fieldnames
        if(fieldnames):
            self._res = self._res/tuple([util.valid_name(fieldname) for fieldname in fieldnames])

        #create representor object
        self._data = data
        tslices = self._res._active_slices
        all_slices = dict([(slice.id, slice) for slice in tslices])
        wrapper.SourceRepresentor.__init__(self, all_slices, tslices)

    def pyexec(self, executor):
        res = self._res._getResult()
        return Result(res)



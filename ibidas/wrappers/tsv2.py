import os
import csv
import sys

csv.field_size_limit(2**31-1)
from logging import info, warning

import wrapper
from .. import ops
from ..constants import *

import python
from ..itypes import detector, rtypes, dimpaths, dimensions
from ..utils import nested_array, util
import numpy
import StringIO
#dialect:
#- delimiter
#- doublequote
#- escapechar
#linterminator
##quotechar

possible_delimiters = numpy.array(['\t','|',':',',','.','$','%','&','#','~','\\','/',';','_','-','+','=','<','>','!','@','^','*'],dtype=object)
possible_quotechars = numpy.array(['"','\''],dtype=object)
possible_escapechars = numpy.array(['\\'],dtype=object)
possible_commentchars = numpy.array(['#','/','*','%','@','!','~','?','|','>'],dtype=object)


class TSVRepresentor(wrapper.SourceRepresentor):
    def __init__(self, filename, skiprows=None, dtype=rtypes.unknown, fieldnames=None, 
                  delimiter=None, quotechar=None, escapechar=None, commentchar=None, skipinitialspace=None, doublequote=None, verbose=True,scan=True):
        
        file = util.open_file(filename,mode='rU')
       
        sample_lines = self.get_sample(file,length=500)
 
        dialect, sniff_message = self.sniff(sample_lines, delimiter=delimiter, quotechar=quotechar, escapechar=escapechar, commentchar=commentchar, skipinitialspace=skipinitialspace, doublequote=doublequote, skiprows=skiprows)

        if verbose:
            print "Determined parameters for file %s:" % filename
            print sniff_message,

        #find sample to determine field names
        sample_lines = self.get_sample(file, length=25, skiprows=dialect.skiprows, commentchar=dialect.commentchar)

        #find start position
        file.seek(0)
        comments = []
        for i in xrange(dialect.skiprows):
            comments.append(file.readline())
        if skiprows is None and dialect.skiprows > 0: #last comment line could be fieldnames
            possible_fieldnames = comments[-1].rstrip('\r\n')
        else:
            possible_fieldnames = ""
        startpos = file.tell()

        data = None
        #determine type
        if(dtype is None):
            dtype = rtypes.unknown
        elif(dtype == rtypes.unknown):
            if(fieldnames is None):
                if len(possible_fieldnames) > 0 and possible_fieldnames[0] == dialect.commentchar:
                    score1 = self.has_header([possible_fieldnames[1:]] + ["\n".join(sample_lines)],dialect)
                else:
                    score1 = -1

                score2 = self.has_header("\n".join(sample_lines), dialect)
                if score1 > score2 and score1 > 0:
                    fieldnames = possible_fieldnames[1:]
                    fieldnames = csv.reader([fieldnames],dialect=dialect).next()
                    comments.pop()
                elif score2 >= score1 and score2 > 0:
                    file.seek(startpos)
                    fieldnames = file.readline()
                    fieldnames = csv.reader([fieldnames],dialect=dialect).next()

            elif(fieldnames is True):
                fieldnames = file.readline()
                fieldnames = csv.reader([fieldnames],dialect=dialect).next()
            
            if(fieldnames):
                fieldnames = [util.valid_name(fieldname) for fieldname in fieldnames]
                if verbose:
                    print '- fieldnames:\t' + ', '.join(fieldnames)
            elif verbose:
                    print '- fieldnames:\tFalse'

            #new start pos
            startpos = file.tell()
            reader = csv.reader(file, dialect)
            #parse data
            if dialect.commentchar:
                commentchar = dialect.commentchar
                data = [tuple(row) for row in reader if len(row) > 0 and not row[0][0] == commentchar]
            else:
                data = [tuple(row) for row in reader if len(row) > 0]
            file.seek(startpos)

            if scan is False or len(data) == 0:
                if fieldnames:
                    nfield = len(fieldnames)
                elif len(data) > 0:
                    z = [len(row) for row in data]
                    nfield = max(z)
                    minz = min(z)
                    fieldnames = ['f' + str(i) for i in xrange(nfield)]
                else:
                    nfield = 0

                if nfield > 0:
                    any = rtypes.TypeAny()
                    anym = rtypes.TypeAny(has_missing=True)
                    ndim = dimensions.Dim(0)
                    subtypes = (any,) * nfield + (anym,) * (nfield - minz)
                    subtype = rtypes.TypeTuple(subtypes=subtypes, fieldnames=fieldnames)
                    rarray = rtypes.TypeArray(subtypes=(subtype,), dims=dimpaths.DimPath(ndim))
                    dtype =rarray
                else:
                    dtype = rtypes.unknown
                
            else:
                det = detector.Detector()
                det.process(data)
                dtype = det.getType()
                if(not fieldnames is None and not fieldnames is False and dtype != rtypes.unknown):
                    assert isinstance(dtype,rtypes.TypeArray),"Error while determining type"
                    assert isinstance(dtype.subtypes[0],rtypes.TypeTuple),"Error while determining type"
                    dtype.subtypes[0].fieldnames = tuple(fieldnames)
        elif(isinstance(dtype,basestring)):
            dtype = rtypes.createType(dtype)
            if(fieldnames is True):
                fieldnames = file.readline()
                fieldnames = csv.reader([fieldnames],dialect=dialect).next()
                startpos = file.tell()
                if verbose:
                    print '- fieldnames:\t' + ', '.join(fieldnames)
            elif verbose:
                    print '- fieldnames:\tFalse'

        else:
            raise RuntimeError, "Not a valid type specified"

        if comments and verbose:
            print 'Comments:'
            print "".join(comments)
    
        slice = TSVOp(filename, dialect, startpos, dtype, "data", lines=data)
        if(slice.type.__class__ is rtypes.TypeArray):
            slice = ops.UnpackArrayOp(slice)
        if(slice.type.__class__ is rtypes.TypeTuple):
            nslices = [ops.UnpackTupleOp(slice, idx) for idx in range(len(slice.type.subtypes))]
        else:
            nslices = [slice]
 
        file.close()
        self._initialize(tuple(nslices))


    def get_sample(self, file, skiprows=0, commentchar='', length=25):
        file.seek(0)
        
        for i in xrange(skiprows):
            file.readline()
         
        sample_lines = []
        for line in file:
            if len(line) == 0 or line[0] == commentchar:
                continue
            sample_lines.append(line.rstrip('\r\n'))
            if(len(sample_lines) > length):
                break
        return sample_lines
    
    def sniff(self, sample_lines, delimiter, quotechar, escapechar, commentchar, skipinitialspace, skiprows, doublequote):
        message = ''
        if commentchar is None:
            fsample_lines = [sl for sl in sample_lines if len(sl) > 0]
            poscomchars = set(possible_commentchars)
            if fsample_lines[0][0] in poscomchars:
                commentchar = fsample_lines[0][0]
            else:
                x = set([sl[0] for sl in fsample_lines if sl[0] in poscomchars])
                if len(x)== 1:
                    commentchar = x.pop()
                elif len(x) >= 1:
                    warning('Multiple possible comment characters: %s. Assuming default commentchar #.' % str(x))
                    commentchar = '#'
                else:
                    commentchar = '#' #assume most often used default
       
        message = '- commentchar:\t' + str(commentchar) + '\n' #+ ('\t!\n' if commentchar is None else '\n')
        if not skiprows is None:
            sample_lines = sample_lines[skiprows:]
            rskiprows = skiprows
        else:
            rskiprows = 0
            for line in sample_lines:
                if len(line) == 0 or line[0] == commentchar:
                    rskiprows += 1
                else:
                    break
            sample_lines = sample_lines[rskiprows:]
            
        message += '- skiprows:\t' + str(rskiprows) + '\n' #+ ('\t!\n' if skiprows >= 0 else '\n')
        sample_lines = [sl for sl in sample_lines if len(sl) == 0 or sl[0] != commentchar]
        
        class sniff_dialect(csv.Dialect):
            _name='sniffed'
            lineterminator = '\r\n'
            quoting = csv.QUOTE_MINIMAL

        if not sample_lines:
            warning('No data, cannot sniff file format! Assuming standard tsv format')
            sniff_dialect.doublequote=doublequote
            sniff_dialect.delimiter=delimiter
            sniff_dialect.quotechar=quotechar or '"'
            sniff_dialect.skipinitialspace=skipinitialspace
            if not escapechar is None:
                sniff_dialect.escapechar=escapechar
            sniff_dialect.commentchar = commentchar
            sniff_dialect.skiprows = rskiprows
            if quotechar is None:
                sniff_dialect.quotechar = '"'
            if delimiter is None:
                sniff_dialect.delimiter = '\t'
            if skipinitialspace is None:
                sniff_dialect.skipinitialspace = True
            if doublequote is None and escapechar is None:
                sniff_dialect.doublequote=True

        else:

            sniff_dialect.doublequote=doublequote
            sniff_dialect.delimiter=delimiter
            sniff_dialect.quotechar=quotechar or '"'
            sniff_dialect.skipinitialspace=skipinitialspace
            if not escapechar is None:
                sniff_dialect.escapechar=escapechar
            sniff_dialect.commentchar=commentchar
            sniff_dialect.skiprows = rskiprows


            if quotechar is None or delimiter is None or skipinitialspace is None:
                if delimiter is None:
                    delimiters = ''.join(possible_delimiters)
                else:
                    delimiters = delimiter

                try:
                    dialect = csv.Sniffer().sniff('\n'.join(sample_lines), delimiters=delimiters)
                
                    if quotechar is None:
                        sniff_dialect.quotechar = dialect.quotechar
                    if delimiter is None:
                        sniff_dialect.delimiter = dialect.delimiter
                    if skipinitialspace is None:
                        sniff_dialect.skipinitialspace = dialect.skipinitialspace
                    if doublequote is None:
                        sniff_dialect.doublequote = dialect.doublequote
                    if not escapechar is None:
                        sniff_dialect.doublequote=False
                except csv.Error:
                    if quotechar is None:
                        sniff_dialect.quotechar = '"'
                    if delimiter is None:
                        sniff_dialect.delimiter = '\t'
                    if skipinitialspace is None:
                        sniff_dialect.skipinitialspace = True
                    if doublequote is None and escapechar is None:
                        sniff_dialect.doublequote=True

        message += '- delimiter:\t' + repr(sniff_dialect.delimiter) + '\n' #+ '\t!\n' if delimiter is None else '\n'
        message += '- quotechar:\t' + sniff_dialect.quotechar +'\n' #+ '\t!\n' if quotechar is None else '\n'
        message += '- skipinitialspace:\t' + str(sniff_dialect.skipinitialspace) +'\n' #+ '\t!\n' if skipinitialspace is None else '\n'
        if sniff_dialect.doublequote:
            message += '- doublequote:\t' + str(sniff_dialect.doublequote)  + '\n' #+ '\t!\n' if doublequote is None else '\n'
        else:
            message += '- escapechar:\t' + str(sniff_dialect.escapechar) + '\n' #+ '\t!\n' if escapechar is None else '\n'

        return (sniff_dialect, message)


    def has_header(self, sample, dialect):
        #copied from python csv module, adapted to add dialect parameter

        # Creates a dictionary of types of data in each column. If any
        # column is of a single type (say, integers), *except* for the first
        # row, then the first row is presumed to be labels. If the type
        # can't be determined, it is assumed to be a string in which case
        # the length of the string is the determining factor: if all of the
        # rows except for the first are the same length, it's a header.
        # Finally, a 'vote' is taken at the end for each column, adding or
        # subtracting from the likelihood of the first row being a header.
        if len(sample) == 0:
            return -1

        rdr = csv.reader(StringIO.StringIO(sample), dialect)

        header = rdr.next() # assume first row is header

        columns = len(header)
        columnTypes = {}
        for i in range(columns): columnTypes[i] = None

        checked = 0
        for row in rdr:
            # arbitrary number of rows to check, to keep it sane
            if checked > 20:
                break
            checked += 1

            if len(row) != columns:
                continue # skip rows that have irregular number of columns

            for col in columnTypes.keys():

                for thisType in [int, long, float, complex]:
                    try:
                        thisType(row[col])
                        break
                    except (ValueError, OverflowError):
                        pass
                else:
                    # fallback to length of string
                    thisType = len(row[col])

                # treat longs as ints
                if thisType == long:
                    thisType = int

                if thisType != columnTypes[col]:
                    if columnTypes[col] is None: # add new column type
                        columnTypes[col] = thisType
                    else:
                        # type is inconsistent, remove column from
                        # consideration
                        del columnTypes[col]

        # finally, compare results against first row and "vote"
        # on whether it's a header
        hasHeader = 0
        for col, colType in columnTypes.items():
            if type(colType) == type(0): # it's a length
                if len(header[col]) != colType:
                    hasHeader += 1
                else:
                    hasHeader -= 1
            else: # attempt typecast
                try:
                    colType(header[col])
                except (ValueError, TypeError):
                    hasHeader += 1
                else:
                    hasHeader -= 1

        return hasHeader


class TSVOp(ops.ExtendOp):
    __slots__ = ["filename", "dialect","startpos"]

    def __init__(self, filename, dialect, startpos, rtype, name, lines=None):
        self.filename = filename
        self.dialect = dialect
        self.startpos = startpos
        self.lines = lines
        ops.ExtendOp.__init__(self,name=name,rtype=rtype)

    def py_exec(self):
        if self.lines is None:
            file = util.open_file(self.filename,mode='rU')
            file.seek(self.startpos)

            if(isinstance(self.dialect, str)):
                reader = csv.reader(file, delimiter=self.dialect)
            else:
                reader = csv.reader(file, self.dialect)
            
            if self.dialect.commentchar:
                commentchar = self.dialect.commentchar
                data = [tuple(row) for row in reader if len(row) > 0 and not row[0][0] == commentchar]
            else:
                data = [tuple(row) for row in reader if len(row) > 0]
            
            file.close()
        else:
            data = self.lines 
       
        ndata = nested_array.NestedArray(data,self.type)
        return python.ResultOp.from_slice(ndata,self)


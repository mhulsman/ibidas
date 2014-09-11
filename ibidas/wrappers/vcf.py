import re
from ..constants import *
from .. utils import util, nested_array
from .. utils.missing import Missing
from ..itypes import detector, rtypes, dimpaths, dimensions

import wrapper
from .. import ops
import python
import numpy

class VCFRepresentor(wrapper.SourceRepresentor):
    def __init__(self, filename):
        self.parser = VCFParser(filename)

        fieldnames = self.parser.fieldNames()
        shapes = self.parser.shapes()
        types = self.parser.types()
        hasmissing = self.parser.hasMissing()
        dimnames = util.gen_seq_names()

        cdimensions = {}
        
        
        #add samples dimension / slice
        cdimensions['samples'] = dimensions.Dim(shape=len(self.parser.sample_names),name='samples')
        sdim = cdimensions['samples']
        subtype = rtypes.createType(str)
        rtype2 = rtypes.TypeArray(dims=dimpaths.DimPath(sdim),subtypes=(subtype,))

        #add variants
        subtypes = []
        for hasmissing, shape, type in zip(hasmissing, shapes,types):
            subtype = rtypes.createType(type)
            if hasmissing:
                subtype.has_missing = True

            if not isinstance(shape,tuple):
                shape = (shape,)

            for pos, s in zip(range(len(shape))[::-1], shape[::-1]):
                if isinstance(s, int):
                    if s <= 1:
                        continue
                    dim = dimensions.Dim(shape=s, name=dimnames.next())
                else:
                    s = s.lower()
                    if not s in cdimensions:
                        cdimensions[s] = dimensions.Dim(shape=UNDEFINED, dependent=(True,) * (pos + 1), name=s)
                    dim = cdimensions[s]
                subtype = rtypes.TypeArray(dims=dimpaths.DimPath(dim),subtypes=(subtype,))
            subtypes.append(subtype)
        

        rtype = rtypes.TypeTuple(subtypes=tuple(subtypes), fieldnames=tuple(fieldnames))
        dim = dimensions.Dim(shape=UNDEFINED, name='variants')
        rtype = rtypes.TypeArray(dims=dimpaths.DimPath(dim), subtypes=(rtype,))
        
        #combine samples slice and variant slices
        alltype = rtypes.TypeTuple(subtypes=(rtype, rtype2), fieldnames=('variants','samples'))
      
        #unpack
        slice = VCFOp(self.parser, alltype, 'vcf')
        slices = [ops.UnpackTupleOp(slice, idx) for idx in range(len(slice.type.subtypes))]
        variant_slice = ops.UnpackArrayOp(slices[0])
        
        all_slices = [ops.UnpackTupleOp(variant_slice, idx) for idx in range(len(variant_slice.type.subtypes))]
        all_slices.append(slices[1])
        for pos, nslice in enumerate(all_slices):
            while(nslice.type.__class__ is rtypes.TypeArray):
                nslice = ops.UnpackArrayOp(nslice)
            all_slices[pos] = nslice
        self._initialize(tuple(all_slices))

class VCFOp(ops.ExtendOp):
    __slots__ = ['parser']
    def __init__(self, parser, rtype, name):
        self.parser = parser
        ops.ExtendOp.__init__(self, name=name, rtype=rtype)

    def py_exec(self):
        data = [row for row in self.parser]
        ndata = nested_array.NestedArray((data, self.parser.sample_names), self.type)
        return python.ResultOp.from_slice(ndata, self)

class VCFParser(object):
    accepted_formats = set(['VCFv4.1'])
    def __init__(self, filename):
        self.filename = filename
        file = util.open_file(filename, mode='rU')
        firstline = file.readline()
        
        version_match = re.match('^\#\#fileformat=(?P<version>.+)$',firstline)
        if version_match is None:
            raise RuntimeError, 'Unexepcted header found in vcf file %s' % self.filename
        version = version_match.group('version')

        if not version in self.accepted_formats:
            util.warning('Unsupported VCF file version: %s. Parser may fail silently!' % version)

        info_pattern = re.compile('^\#\#INFO=\<ID=(?P<name>[^,]+),Number=(?P<number>[\d]+|A|G|\.),Type=(?P<type>Integer|Float|Flag|Character|String),Description=\"(?P<description>[^\"]*)\"\>$')
        format_pattern = re.compile('^\#\#FORMAT=\<ID=(?P<name>[^,]+),Number=(?P<number>[\d]+|A|G|\.),Type=(?P<type>Integer|Float|Character|String),Description=\"(?P<description>[^\"]*)\"\>$')

        info_fields = []
        format_fields = []

        for line in file:       
            if line.startswith('##INFO'):
                res = info_pattern.match(line)
                if res is None:
                    raise RuntimeError, 'INFO line does not conform to specification: %s' % line
                number = self._numberConvert(res.group('name'),res.group('number'))
                xtype = self._toType(res.group('type'))
                info_fields.append((res.group('name'), number, xtype, self._toDType(xtype, number), res.group('description')))
            elif line.startswith('##FORMAT'):
                res = format_pattern.match(line)
                if res is None:
                    raise RuntimeError, 'FORMAT line does not conform to specification: %s' % line
                number = self._numberConvert(res.group('name'),res.group('number'))
                xtype = self._toType(res.group('type'))
                format_fields.append((res.group('name'), number, xtype, self._toDType(xtype, number), res.group('description')))
            elif line.startswith("#CHROM"):
                self.sample_names = line.strip().split('\t')[9:]
            elif not line.startswith('#'):
                break
        file.close()                

        self.info_fields = info_fields
        self.format_fields = format_fields
        self.has_genotypes = len(self.sample_names) > 0

  
    def fieldNames(self):
        names = ('chrom','pos', 'id', 'ref','alt','qual','filter',)
        inames = tuple([name.lower() for name, number, type, dtype, description in self.info_fields])
        fnames = tuple([name.lower() for name, number, type, dtype, description in self.format_fields])
        return names + inames + fnames

    def _numberConvert(self, name, number):
            if number == 'A':
                return 'alt_alleles'
            elif number == 'G':
                return 'genotypes'
            elif number == '.':
                return name
            else:
                try:
                    return int(number)
                except ValueError:
                    raise RuntimeError, 'Unexpected number format: %s' %number

    def shapes(self):
        shapes = (1,1,'ids',1,'alt_alleles',1,'filters',)

        ishapes = [number for name, number, type, dtype, description in self.info_fields]
        fshapes = [('samples', number) for name, number, type, dtype, description in self.format_fields]
       
        return shapes + tuple(ishapes) + tuple(fshapes)

    def types(self):
        types = (str, int, str, 'DNA', 'DNA', float, str,)
        itypes = [type for name, number, type, dtype, description in self.info_fields]
        ftypes = [type for name, number, type, dtype, description in self.format_fields]

        return types + tuple(itypes) + tuple(ftypes)

    def hasMissing(self):
        hasmissing = (False, False, False, False, False, True, False,)
        ihasmissing = [isinstance(number, int) or number in set(['alt_alleles','genotypes']) for name, number, type, dtype, description in self.info_fields]
        fhasmissing = [isinstance(number, int) or number in set(['alt_alleles','genotypes']) for name, number, type, dtype, description in self.format_fields]
        return hasmissing + tuple(ihasmissing) + tuple(fhasmissing)

    def _toType(self, typename):
        if typename == 'Integer':
            return int
        elif typename == 'String':
            return str
        elif typename == 'Float':
            return float
        elif typename == 'Flag':
            return bool
        elif typename == 'Character':
            return str
        else:
            raise RuntimeError,'Unknown type %s' % typename

    def processLine(self, line):
        line = line.strip()
        elems =  line.split('\t')
        chrom, pos, id, ref, alt, qual, filter, info = elems[:8]
        
        pos = int(pos)
        id = '' if id == '.' else id.split(';')
        alt = alt.split(','); #FIXME
        qual = Missing if qual == '.' else 10.0**(float(qual)/-10.0)
        filter = [] if filter == '.' or filter == 'PASS' else filter.split(',')

        dinfo = {}
        for elem in info.split(';'):
            if not '=' in elem:
                dinfo[elem] = True
            else:                
                key,value = elem.split('=',1)
                dinfo[key] = value

        
        xinfo = []
        for name, number, type, dtype, description in self.info_fields:
            if name in dinfo:
                if number == 1 or number == 0:
                    xinfo.append(type(dinfo[name]))
                else:
                    xinfo.append([type(elem) for elem in dinfo[name].split(',')])
            else:
                xinfo.append(self._genMissing(number, alt, dtype))
        
        xformat = []
        if self.has_genotypes:
            format_fields = elems[8].split(':')
            format = elems[9:]

            format = zip(*[elem.split(':') for elem in format])
            dformat = dict([(ffield,list(values)) for ffield, values in zip(format_fields, format)])


            for name, number, type, dtype, description in self.format_fields:
                #func = lambda x: type(x) if x.strip() != '.' else Missing
                if name in dformat:
                    if number == 1:
                        xformat.append([type(elem) if elem.strip() != '.' else Missing for elem in dformat[name]])
                    else:
                        xformat.append([[type(elem) for elem in xelem.split(',')] if xelem != '.' else self._genMissing(number, alt, dtype) for xelem in dformat[name]])
                else:
                    xformat.append([self._genMissing(number, alt, dtype)] * len(elems[9:]))

        return (chrom, pos, id, ref, alt, qual, filter,) + tuple(xinfo) + tuple(xformat)
        

    def _toDType(self, type, number):
        if isinstance(type,str) or isinstance(number, int) or number == 'alt_alleles' or number == 'genotypes':
            return object
        else: 
            return numpy.dtype(type)

    def _genMissing(self, number, alt, dtype=None):
        if number == 0 and dtype is bool:
            res = False
        elif number == 1:
            res = Missing
        elif isinstance(number,int):
            res = [Missing] * number
        elif number == 'alt_alleles':
            res = [Missing] * len(alt)
        elif number == 'genotypes':
            res = [Missing] * (((len(alt) + 1) * (len(alt) + 2)) / 2)
        else:
            res = []
        return res


    def __iter__(self):
        file = util.open_file(self.filename, mode='rU')

        for line in file:
            if line.startswith('#'):
                continue
            else:
                yield self.processLine(line)

        for line in file:
            yield self.processLine(line)
        
        file.close()

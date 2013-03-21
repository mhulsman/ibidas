from itertools import chain
import re
from ..constants import *
from .. utils import util, nested_array
from .. utils.missing import Missing
from ..itypes import detector, rtypes, dimpaths, dimensions
from .. import repops_multi
import wrapper
from .. import ops
import python
from pyparsing import *


data_class = {'CON':'contig','PAT':'patent', 'EST':'expressed sequence tag',
              'GSS':'genome survey sequence','HTC':'high thoughput CDNA sequencing',
              'HTG':'high thoughput genome sequencing', 'MGA':'mass genome annotation',
              'WGS':'whole genome shotgun', 'TSA':'transcriptome shotgun assembly',
              'STS':'sequence tagged site', 'STD':'standard'}

taxonomy = {'PHG':'Bacteriophage',       
            'ENV':'Environmental Sample',
            'FUN':'Fungal',              
            'HUM':'Human',               
            'INV':'Invertebrate',        
            'MAM':'Other Mammal',        
            'VRT':'Other Vertebrate',    
            'MUS':'Mus musculus',         
            'PLN':'Plant',               
            'PRO':'Prokaryote',          
            'ROD':'Other Rodent',        
            'SYN':'Synthetic',           
            'TGN':'Transgenic',          
            'UNC':'Unclassified',        
            'VRL':'Viral'}               


class GERepresentor(wrapper.SourceRepresentor):
    def __init__(self, filename, type):
        if type == 'embl':
            parser = EMBLParser(filename)
        elif type == 'genbank':
            parser = GenbankParser(filename)
        
        feats_dim = dimensions.Dim(name='feats',shape=UNDEFINED, dependent=(True,))    
        records_dim = dimensions.Dim(name='records',shape=UNDEFINED)

        records = [r for r in parser]

        seq = self._getSequenceRep([r.get('sequence',Missing) for r in records], records_dim)
        loc_type = self._getLocTypeRep([r.get('feat_loc_type') for r in records], feats_dim, records_dim)
        loc = self._getLocsRep([r.get('feat_locs') for r in records], feats_dim, records_dim)
        rec = self._getRecordRep([r.get('record') for r in records], records_dim)
        attr = self._getAttrRep([r.get('feat_attr') for r in records], feats_dim, records_dim)

        res = repops_multi.Combine(rec, loc_type, loc, attr, seq)

        self._initialize(tuple(res._slices))

    rectypes = {'length':int, 'alt_accessions':('alt_accessions',str), 'keywords':('keywords',str), 'organism_class':('organism_class',str), 'database':('references',str), 'database_identifier':('references',str)}

    def _getRecordRep(self, records, records_dim):
        keys = list(set(chain(*[r.keys() for r in records])))
        keys.sort()
        types = []
        dims = {}
        for key in keys:
            if key in self.rectypes:
                t = self.rectypes[key]
                if isinstance(t, tuple):
                    s = rtypes.createType(t[-1])
                    for d in t[::-1][1:]:
                        if not d in dims:
                            dim = dimensions.Dim(name=d,shape=UNDEFINED, dependent=(True,))
                            dims[d] = dim
                        s = rtypes.TypeArray(dims=dimpaths.DimPath(dims[d]), subtypes=(s,))
                    t = s
                else:
                    t = rtypes.createType(t)
            else:
                t = rtypes.createType(str)
            types.append(t)

        rectype = rtypes.TypeRecordDict(fieldnames=tuple(keys), subtypes=tuple(types))
        rectype = rtypes.TypeArray(dims=dimpaths.DimPath(records_dim), subtypes=(rectype,))
        return python.Rep(records, dtype=rectype)

    attrtypes = {'transl_table':int, 'xref_database': ('feat_ref',str), 'xref_identifier' : ('feat_ref', str),
                'codon_start': int, 'environmental_sample':bool}
                
    def _getAttrRep(self, records, feats_dim, records_dim):
        types = python.Rep(records,unpack=False).Type
        attrtype = types.subtypes[0].subtypes[0]
        attrtype = rtypes.TypeArray(dims=dimpaths.DimPath(feats_dim), subtypes=(attrtype,))
        attrtype = rtypes.TypeArray(dims=dimpaths.DimPath(records_dim), subtypes=(attrtype,))
        return python.Rep(records, dtype=attrtype)

    def _getSequenceRep(self, seq, records_dim):
        dna = rtypes.createType('DNA')
        seqtype = rtypes.TypeArray(dims=dimpaths.DimPath(records_dim), subtypes=(dna,))
        return python.Rep(seq,dtype=seqtype, name='sequence')

    def _getLocsRep(self, data, feats_dim, records_dim):
        names = ['name','start','stop','complement','type','fuzzy_before','fuzzy_after']
        types = [str, int, int, bool, str, bool, bool]
        types = [rtypes.createType(t) for t in types]
        loctype = rtypes.TypeRecordDict(fieldnames=tuple(names), subtypes=tuple(types))
        dim = dimensions.Dim(name='loc_elems',shape=UNDEFINED, dependent=(True,))
        loctype = rtypes.TypeArray(dims=dimpaths.DimPath(dim), subtypes=(loctype,))
        loctype = rtypes.TypeArray(dims=dimpaths.DimPath(feats_dim), subtypes=(loctype,))
        loctype = rtypes.TypeArray(dims=dimpaths.DimPath(records_dim), subtypes=(loctype,))
        return python.Rep(data,dtype=loctype)

    def _getLocTypeRep(self, data, feats_dim, records_dim):
        loctype = rtypes.createType(str)
        loctype = rtypes.TypeArray(dims=dimpaths.DimPath(feats_dim), subtypes=(loctype,))
        loctype = rtypes.TypeArray(dims=dimpaths.DimPath(records_dim), subtypes=(loctype,))
        return python.Rep(data,dtype=loctype,name='loc_type')
       


featrec = ZeroOrMore(Group((LineStart() | Word('\n')) +  Suppress('/') + Word(alphas + '_') + Optional(Suppress('=') + (Word(nums).setParseAction(lambda x: map(int,x)) | QuotedString(quoteChar='"',  escQuote='""', multiline=True).setParseAction(lambda x: [e.replace('\n','') for e in x])))))

num = Word(nums)
date = num + Suppress('-') + Word(alphas) + Suppress('-') + num 
date_created = date + Suppress('(Rel.') + num + Suppress(', Created)')
date_updated = date + Suppress('(Rel.') + num + Suppress(', Last updated, Version') + num + Suppress(')')
organism = Group(OneOrMore(Word(alphas))) + Group(Optional('(' + OneOrMore(Word(alphas)) + ')'))


#location
number = Word(nums).setParseAction(lambda x: map(int,x))
name = CharsNotIn(':')
identifier1 =  (Optional('<').setResultsName('fuzzybefore') + number.setResultsName('start'))
identifier2 = (Optional('>').setResultsName('fuzzyafter') + number.setResultsName('stop'))
loc = Group(Optional(name.setResultsName('name') + ':') + (identifier1 + oneOf('.. . ^').setResultsName('operator') + identifier2 | number.setResultsName('start')))

loc_expression = Forward()
lparen = Literal("(").suppress()
rparen = Literal(")").suppress()

arg = loc | loc_expression
args = delimitedList(arg)
functor = (Literal('complement') | Literal('join') | Literal('order')).setResultsName('function')
loc_expression << (Group(functor + Group(lparen + Optional(args) + rparen)) | loc)



class GEParser(object):
    types = {}
    dims = {}

    def __init__(self, filename):
        self.reader = util.PeekAheadFileReader(filename)
        self.skipcache = set()

    def __iter__(self):
        return self
    
    def next(self): 
        return self.parseRecord()

    def startRecord(self):
        self.record = {}
        self.record['record'] = {}

    def finishRecord(self):
        return self.record

    def parseRecord(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def next(self):
        r = self.parseRecord()
        if r is None:
            raise StopIteration
        return r

    def parseFeatures(self, format='embl'):
        if format == 'embl':
            check = set(['FT','XX'])
        else:
            check == set([''])
        rc = self.record

        feat_key = []
        feat_loc_type = []
        feat_locs = []
        feat_attr = []
        valid_name = util.valid_name

        for line in self.reader:
            lineNr = self.reader.lineNr
            if not line[:5].strip() in check:
                self.reader.pushBack()
                break
            key = line[5:20].rstrip()
            data = [line[21:].rstrip()]
            
            while(self.reader.peekAhead()[:20].strip() in check):
                data.append(self.reader.next()[21:].rstrip())
            
            loc = data[0]
            try:
                loc_results = loc_expression.parseString(loc)
                outer, inner = self._loc_process(loc_results)
                if outer is None:
                    if len(inner) > 1:
                        util.warning('Order/join attribute not specified at line %d for list of locations "%s", assuming join.' %(lineNr, loc))
                    outer = 'join'
                feat_loc_type.append(outer)
                feat_locs.append(inner)
            except ParseException, e:
                util.warning('Failed to parse location "%s" for record starting at line number: %d\n' % (loc, lineNr))
                continue

            attributes = '\n'.join(data[1:])
            try:
                results = featrec.parseString(attributes)
                nresults = {}
                for e in results:
                    name = valid_name(e[0])
                    func = getattr(self, 'attr' + name, self.attrElse)
                    if len(e) > 1:
                        value = e[1]
                    else:
                        value = True
                    func(nresults, name, value)
                results = nresults
            except ParseException, e:
                feature = key + ' @ ' + loc + '\n'  +  attributes
                util.warning('Failed to parse attributes for record starting at line number: %d\nFeature:\n%s\n' % (lineNr, feature))
                continue
           
            feat_key.append(key)
            feat_attr.append(results)

        rc['feat_key'] = feat_key
        rc['feat_loc_type'] = feat_loc_type
        rc['feat_locs'] = feat_locs
        rc['feat_attr'] = feat_attr

    def attrElse(self, res, key, value):
        res[key] = value

    def attrMulti(self, res, key, value):
        if key in res:
            res[key].append(value)
        else:
            res[key] = [value]

    def attrdb_xref(self, res, key, value):
        if ',' in value:
            values = [e.strip().split(':') for e in value.split(',')]
            databases = [e[0] for e in values]
            identifiers = [e[1] for e in values]
            if 'xref_database' in res:
                res['xref_database'].extend(databases)
                res['xref_identifier'].extend(identifiers)
            else:
                res['xref_database'] = databases
                res['xref_identifier'] = identifiers
        else:
            database, identifier = value.split(':')
            if 'xref_database' in res:
                res['xref_database'].append(database)
                res['xref_identifier'].append(identifier)
            else:
                res['xref_database'] = [database]
                res['xref_identifier'] = [identifier]

    def _loc_process(self, loc_results):
        if len(loc_results) == 1:
            loc_results = loc_results[0]
        if 'function' in loc_results:
            res = [self._loc_process(e) for e in loc_results[1]]
            
            if not res:
                util.warning('Location function without argument at line %d, skipping.', self.reader.lineNr)
                return ('join',[])
           
            ninners= []
            nouters = []
            for outer, inner in res:
                ninners.extend(inner)
                nouters.append(outer)
           
            if loc_results['function'] == 'complement':
                for inner in ninners:
                    inner['complement'] = not inner['complement']
            elif loc_results['function'] == 'join':
                nouters.append('join')
            elif loc_results['function'] == 'order':
                nouters.append('order')
            else:
                util.warning('Unknown function used at line %d, skipping.', self.reader.lineNr)
            
            nouters = set(nouters)
            if len(nouters) > 1:
                nouters.discard(None)
            if len(nouters) > 1:
                util.warning('Both join and order attributes are used simultaneously for record on line %d. This is illegal, assuming "join".', self.reader.lineNr)
                nouters = set(['join'])
            outer = nouters.pop()
            

            return (outer, ninners)
        else:
           nres = {}
           nres['name'] = loc_results.get('name', '')
           nres['start'] = loc_results['start']
           nres['stop'] = loc_results.get('stop', nres['start'])
           nres['complement'] = False
           
           between = loc_results.get('operator','') == '^'
           single_base = nres['start'] == nres['stop'] or loc_results.get('operator','') == '.'
           if between:
               t = 'between_base'
           elif single_base:
               t = 'single_base'
           else:
               t = 'region'
           nres['type'] = t

           nres['fuzzy_before'] = loc_results.get('fuzzybefore', '') == '<'
           nres['fuzzy_after'] = loc_results.get('fuzzyafter', '') == '>'
           return (None, [nres])
        


class EMBLParser(GEParser):
    def parseRecord(self):
        if self.reader.eof():
            return None

        self.startRecord()
        pos = self.reader.tell()

        #skip possible header comments
        try:
            while not self.reader.peekAhead()[:2] == 'ID':
                self.reader.next() 
        except StopIteration,e:
            util.warning('Arrived at end of file without ever seeing an ID record. Restarting parsing attempt without looking for ID record.')
            self.reader.reset(pos)
            try:
                while not len(self.reader.peekAhead()[:5].rstrip()) == 2:
                    self.reader.next()
            except StopIteration, e:
                 util.error('No parseable lines found, this is not a proper EMBL file.')
        
        #start parsing
        for line in self.reader:
            code = line[:2]
            if code == '//':
                return self.finishRecord()
            line = line[5:].rstrip()
            getattr(self, 'embl' + code, self.emblElse)(code, line)
        
        util.warning('Unexpected end of file. Record not closed by '//'')
        return self.finishRecord()


    def emblElse(self, code, line):
        if not code in self.skipcache:
            util.warning('Skipping unknown header code "%s" at line %d (subsequent occurences will not be reported). Line:\n%s\n' % (code, self.reader.lineNr, self.reader.curLine.strip()))
            self.skipcache.add(code)

    def emblID(self, code, line):
        cols = [e.strip() for e in line.split(';')]
        r = self.record['record']
        length = None

        if len(cols) == 7: #assuming newer version of ID line 
            r['accession'] = cols[0] if cols[0] else Missing
            r['version'] = cols[1] if cols[1] else Missing
            r['topology'] = cols[2] if cols[2] else Missing
            r['mol_type'] = cols[3] if cols[3] else Missing
            r['data_class'] = data_class.get(str.upper(cols[4]),cols[4]) if cols[4] else Missing
            r['taxonomy'] = taxonomy.get(str.upper(cols[5]),cols[5]) if cols[5] else Missing
            length = util.getNumber.match(cols[6])
            if cols[6] and length is None:
                util.warning('Unable to parse length: %s', cols[6])
        else:
            if cols:
                accession = cols[0].split()
                if len(accession) == 2:
                    r['accession'] = accession[0] if accession[0] else Missing
                    r['data_class'] = data_class.get(str.upper(accession[1]),accession[1]) if accession[1] else Missing
                else:
                    r['accession'] = accession[0] if accession[0] else Missing
            
            if len(cols) == 4:
                r['molecule_type'] = cols[1] if cols[1] else Missing
                r['taxonomy'] = taxonomy.get(str.upper(cols[2]),cols[2]) if cols[2] else Missing
                length = util.getNumber.match(cols[3])
                if cols[3] and length is None:
                    util.warning('Unable to parse length: %s', cols[3])
            else:
                util.warning('ID line has incorrect format: %s', line)

        if not length is None and len(length.groups()) > 0:
            r['length'] = int(length.groups()[0])

    def emblAC(self, code, line):
        rc = self.record['record']

        acs = [e for e in [e.strip() for e in line.split(';')] if e]
        if not 'alt_accessions' in rc:
           rc['alt_accessions'] = []
        rc['alt_accessions'].extend(acs)

    def emblDE(self, code, line):
        rc = self.record['record']

        if 'description' in rc:
            rc['description'] += ' ' + line
        else:
            rc['description'] = line

    def emblPR(self, code, line):
        self.record['record']['project_identifier'] = line

    def emblKW(self, code, line):
        if line.endswith('.'):
            line = line[:-1]
        self.record['record']['keywords'] = [e.strip() for e in line.split(';')]

    def emblOS(self, code, line):
        rc = self.record['record']
        try: 
            n = organism.parseString(line)
            rc['organism'] = ' '.join(n[0])
            if len(n) > 1 and n[1]:
                rc['organism_common_name'] = ' '.join(n[1])
        except ParseException,e:
            util.warning('Failed to parse organism at line number: %d\nLine:\n%s\n' % (self.reader.lineNr, line))

    def emblOC(self, code, line):
        rc = self.record['record']
        if line.endswith('.'):
            line = line[:-1]
        if not 'organism_class' in rc:
            rc['organism_class'] = []
        rc['organism_class'].extend([e for e in [e.strip() for e in line.split(';')] if e])
            
    def emblOG(self, code, line):
        rc['organelle'] = line               


    #REFERENCES: FIXME
    def emblRN(self, code, line):
        pass
    emblRC = emblRN
    emblRP = emblRN
    emblRX = emblRN
    emblRG = emblRN
    emblRA = emblRN
    emblRT = emblRN
    emblRL = emblRN

    def emblDR(self, code, line):
        rc = self.record['record']
        if line.endswith('.'):
            line = line[:-1]
        line = [e.strip() for e in line.split(';')]
        if not 'database' in rc:
            rc['database'] = []
            rc['database_identifier'] = []
        rc['database'].append(line[0])
        rc['database_identifier'].append(line[1])

    def emblCC(self, code, line):
        rc = self.record['record']
        if not 'comment' in rc:
            rc['comment'] = line    
        else:
            rc['comment'] = rc['comment'] + ' ' + line

    def emblDT(self, code, line):
        rc = self.record['record']
        try:
            if not 'date_created' in rc:
                res = date_created.parseString(line)
                rc['date_created'] = res[0] + '-' + res[1] + '-' + res[2]
            elif not 'date_modified' in rc:
                res = date_updated.parseString(line)
                rc['date_modified'] = res[0] + '-' + res[1] + '-' + res[2]
                rc['version'] = res[4]
            else:
                util.warning('More DT (date) lines then expected at line %d', self.reader.lineNr)
        except ParseException, e:
            util.warning('Failed to parse date at line number: %d\nLine:\n%s\n' % (self.reader.lineNr, line))

    def emblFH(self, code, line):
        pass

    def emblXX(self, code, line):
        pass

    def emblFT(self, code, line):
        self.reader.pushBack()
        self.parseFeatures(format='embl')

    def emblSQ(self, code, line):
        seq = []
        for line in self.reader:
            if not line[:5].strip() == '':
                self.reader.pushBack()
                break
            seq.append(line[5:72].replace(' ',''))
        if seq:
            self.record['sequence'] = ''.join(seq)
        else:
            util.warning('SQ record without sequence?! Will attempt to go on.')




class GenbankParser(GEParser):
    def parseGenbank(self):
        if self.reader.eof():
            return None
        self.startRecord()

        #skip possible header comments
        pos = self.reader.tell()
        try:
            while not self.reader.peekAhead()[:12] == 'LOCUS':
                self.reader.next() 
        except StopIteration,e:
            util.warning('Arrived at end of file without ever seeing an LOCUS record. Restarting parsing attempt without looking for LOCUS record.')
            self.reader.reset()

        for line in self.reader:
            if not line.strip():
                continue
                
            if line[:2] == '//': #end of record
                return self.finishRecord()
                
            code = line[:12].rstrip()
            if code[0] == ' ': #continuation
                if len(code[:6]).lstrip() == 1: #sudden start of feature table?
                     util.warning('Unexpected start of feature table at line %d: %s. Will attempt parsing.', self.reader.lineNr, line)
                     self.reader.pushBack()
                     self.parseFeatures()
                continue
            line = line[12:].rstrip()
            getattr(self, 'genbank' + code, self.genbankElse)(code, line)

        return self.finishRecord()
           
    def genbankElse(self, code, line):
        if not code in self.skipcache:
            util.warning('Skipping unknown header code "%s" at line %d (subsequent occurences will not be reported). Line:\n%s\n' % (code, self.reader.lineNr, self.reader.curLine.strip()))
            self.skipcache.add(code)

    def genbankFEATURES(self, code, line):
        self.parseFeatures(format='genbank')

    def genbankLOCUS(self, code, line):
        pass






import re
from ..constants import *
from .. utils import util, nested_array
from .. utils.missing import Missing
from ..itypes import detector, rtypes, dimpaths, dimensions

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


class GenbankRepresentor(wrapper.SourceRepresentor):
    def __init__(self, filename):
        self.parser = GenbankParser(filename)


featrec = ZeroOrMore(Group((LineStart() | Word('\n')) +  Suppress('/') + Word(alphas + '_') + Suppress('=') + \
          Word(nums).setParseAction(lambda x: map(int,x)) | QuotedString(quoteChar='"', multiline=True).setParseAction(lambda x: [e.replace('\n',' ') for e in x])))

featrec = ZeroOrMore(Group((LineStart() | Word('\n')) +  Suppress('/') + Word(alphas + '_') + Suppress('=') + (Word(nums).setParseAction(lambda x: map(int,x)) | QuotedString(quoteChar='"', multiline=True).setParseAction(lambda x: [e.replace('\n',' ') for e in x]))))

num = Word(nums)
date = num + Suppress('-') + Word(alphas) + Suppress('-') + num 
date_created = date + Suppress('(Rel.') + num + Suppress(', Created)')
date_updated = date + Suppress('(Rel.') + num + Suppress(', Last updated, Version') + num + Suppress(')')
organism = Group(OneOrMore(Word(alphas))) + Group(Optional('(' + OneOrMore(Word(alphas)) + ')'))

class GenbankParser(object):
    types = {}
    dims = {}

    def __init__(self, filename, format='EMBL'):
        self.reader = util.PeekAheadFileReader(filename)
        self.skipcache = set()
        self.format = format

    def __iter__(self):
        return self
    
    def next(self): 
        return self.parseRecord()

    def parseRecord(self):
        if self.reader.eof():
            return None

        if self.format == 'EMBL':
            return self.parseEMBL()
        elif self.format == 'Genbank':
            return self.parseGenbank()
        else:
            util.error('Unknown file format: %s' % self.format)

    def startRecord(self):
        self.record = {}
        self.record['record'] = {}

    def finishRecord(self):
        return self.record

    def parseEMBL(self):
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
            r['data_class'] = data_class.get(cols[4],cols[4]) if cols[4] else Missing
            r['taxonomy'] = taxonomy.get(cols[5],cols[5]) if cols[5] else Missing
            length = util.getNumber.match(cols[6])
            if cols[6] and length is None:
                util.warning('Unable to parse length: %s', cols[6])
        else:
            if cols:
                accession = cols[0].split()
                if len(accession) == 2:
                    r['accession'] = accession[0] if accession[0] else Missing
                    r['data_class'] = data_class.get(accession[1],accession[1]) if accession[1] else Missing
                else:
                    r['accession'] = accession[0] if accession[0] else Missing
            
            if len(cols) == 4:
                r['molecule_type'] = cols[1] if cols[1] else Missing
                r['taxonomy'] = taxonomy.get(cols[2],cols[2]) if cols[2] else Missing
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

    def parseGenbank(self):
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
           
    genbankElse = emblElse

    def genbankFEATURES(self, code, line):
        self.parseFeatures(format='genbank')

    def genbankLOCUS(self, code, line):
        pass

    def parseFeatures(self, format='embl'):
        if format == 'embl':
            check = set(['FT','XX'])
        else:
            check == set([''])

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
            attributes = '\n'.join(data[1:])
            try:
                results = featrec.parseString(attributes)
            except ParsingError, e:
                feature = key + ' @ ' + loc + '\n'  +  attributes
                util.warning('Failed to parse attributes for record starting at line number: %d\nFeature:\n%s\n' % (lineNr, feature))
                continue
            
            getattr(self, 'feat' + key, self.featElse)(key, loc, results, lineNr)


    def featElse(self, key, loc, attributes, lineNr):
        if not key in self.skipcache:
            feature = key + ' @ ' + loc + '\n'  +  '\n'.join([e[0] + '=' + str(e[1]) for e in attributes])
            util.warning('Skipping unknown feature key "%s" at line %d (subsequent occurences will not be reported).\nFeature:\n%s\n' % (key, lineNr, feature))
            self.skipcache.add(key)





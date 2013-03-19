import re
from ..constants import *
from .. utils import util, nested_array
from .. utils.missing import Missing
from ..itypes import detector, rtypes, dimpaths, dimensions

import wrapper
from .. import ops
import python
from pyparsing import *

class GenbankRepresentor(wrapper.SourceRepresentor):
    def __init__(self, filename):
        self.parser = GenbankParser(filename)


featrec = ZeroOrMore(Group((LineStart() | Word('\n')) +  Suppress('/') + Word(alphas + '_') + Suppress('=') + \
          Word(nums).setParseAction(lambda x: map(int,x)) | QuotedString(quoteChar='"', multiline=True).setParseAction(lambda x: [e.replace('\n',' ') for e in x])))

featrec = ZeroOrMore(Group((LineStart() | Word('\n')) +  Suppress('/') + Word(alphas + '_') + Suppress('=') + (Word(nums).setParseAction(lambda x: map(int,x)) | QuotedString(quoteChar='"', multiline=True).setParseAction(lambda x: [e.replace('\n',' ') for e in x]))))


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
        if self.format == 'EMBL':
            return self.parseEMBL()
        elif self.format == 'Genbank':
            return self.parseGenbank()
        else:
            util.error('Unknown file format: %s' % self.format)

    def startRecord(self):
        self.record = {}

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
            line = line[6:].rstrip()
            getattr(self, 'embl' + code, self.emblElse)(code, line)

    def emblElse(self, code, line):
        if not code in self.skipcache:
            util.warning('Skipping unknown header code "%s" at line %d (subsequent occurences will not be reported). Line:\n%s\n' % (code, self.reader.lineNr, self.reader.curLine.strip()))
            self.skipcache.add(code)

    def emblID(self, code, line):
        pass

    def emblFH(self, code, line):
        pass

    def emblXX(self, code, line):
        pass

    def emblFT(self, code, line):
        self.reader.pushBack()
        self.parseFeatures(format='embl')

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





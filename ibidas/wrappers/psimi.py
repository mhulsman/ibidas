"""
Parser for PSI-MI files.
Specification overview: http://dip.doe-mbi.ucla.edu/psimi/MIF254.xsd
"""

import re
from itertools import chain
import sys, string
import xml.sax.handler
import xml.sax
from collections import deque,defaultdict, namedtuple
import xml.parsers
import xml.parsers.expat
import numpy

from ..utils.multi_visitor import DirectVisitorFactory, NF_ELSE
from ..utils.missing import Missing
import wrapper
from ..itypes import rtypes
from .. import ops
from ..constants import *
import python
from ..utils import nested_array,util

def read_psimi(filename):
    return PSIMIRepresentor(filename).Fields().Copy()

Interaction = namedtuple('Interaction', ('id', 'type', 'participants', 'experiments', 'biogrid_type', 'author_confidence', 'intact_confidence'))
interaction_type= "[interactions:*]<(id=int32, type=string?, " + \
                    "participants=[iinteractors:~]<(id=int32, interactor_id=int32, role=string[]?), " + \
                    "experiments=[iexperiemnts:~]<int32, biogrid_type=string[]?, author_confidence=string[]?, intact_confidence=real32?" + \
                    ")"
Participant = namedtuple('Participant', ('id', 'interactor_id', 'role'))

Interactor = namedtuple('Interactor', ('id', 'short_name', 'refseq_id', 'type_id'))
interactor_type = "[interactors:*]<(id=int32, short_name=string[], refseq_id=string[]?, type_id=string[]?)"

Experiment = namedtuple('Experiment', ('id', 'method_id', 'method_name', 'pubmed_id'))
experiment_type = "[experiments:*]<(id=int32, method_id=string[], method_name=string[], pubmed_id=string[])"

Organism = namedtuple('Organism', ('id', 'short_label', 'full_name'))

class PSIMIRepresentor(wrapper.SourceRepresentor):
    def __init__(self, filename):
        xtype = "(experiment=" + experiment_type + ", interactor=" + interactor_type + ", interaction=" + interaction_type + ")"
        xtype = rtypes.createType(xtype)
        slice = PSIMIOp(filename,xtype,"data")
        self._initialize((slice,))



class PSIMIOp(ops.ExtendOp):
    __slots__ = ['filename']
    def __init__(self, filename, rtype, name="data"):
        self.filename = filename
        ops.ExtendOp.__init__(self, name=name, rtype=rtype)

    def py_exec(self):
        file = open(self.filename)
        psimi_handler = PSIMIParser()
        parser = xml.parsers.expat.ParserCreate()
        parser.StartElementHandler = psimi_handler.startElement
        parser.EndElementHandler = psimi_handler.endElement
        parser.CharacterDataHandler = psimi_handler.characters
        #parser.ExternalEntityRefHandler = psimi_handler.resolveEntity
        parser.UseForeignDTD(True)
        parser.buffer_text = True
        parser.ParseFile(file)

        file.close()
        exps = [Experiment(**exp) for exp in psimi_handler.experiments.values()]
        interactors = [Interactor(**inter) for inter in psimi_handler.interactors.values()]
        interactions = []
        for interaction in psimi_handler.interactions.values():
            interaction['participants'] = [Participant(**par) for par in interaction['participants']]
            interactions.append(Interaction(**interaction))

        data = (exps, interactors, interactions)
        ndata = nested_array.NestedArray(data,self.type)
        return python.ResultOp.from_slice(ndata,self)
        


"""
Handler class for actually processing the file
"""    
class PSIMIParser(DirectVisitorFactory(prefixes=("start", "end"), flags=NF_ELSE), xml.sax.handler.ContentHandler):

   def __init__(self):
      self.state = deque()
      self.interactions = {}
      self.cur_interaction = None
      self.interactors = {}
      self.cur_interactor = None
      self.experiments = {}
      self.cur_experiment = None
      self.organisms = {}
      self.cur_organism = None

      self.int_participants = []
      self.cur_participant = None
      self.int_experiments = []


      self.cur_attribute = ""
      self.cur_confidence_type = ""

   def startelse(self, name, attrs):
      pass

   def startinteraction(self, name, attrs):
      self.cur_interaction = {'id': int(attrs['id']), 'author_confidence':Missing, 'intact_confidence':numpy.nan, 'biogrid_type':Missing, 'type':Missing}
      self.int_experiments = []
      if('negative' in attrs):
          #print "Found negative interaction!!"
          raise Exception, 'negative interaction encountered, implement me!'
   
   def startparticipantList(self, name, attrs):
       self.int_participants = []

   def startparticipant(self, name, attrs):
       self.cur_participant = {'id':int(attrs['id']), 'role':Missing}

   def endparticipant(self, name):
       self.int_participants.append(self.cur_participant)
       
   def endparticipantList(self, name):
       self.cur_interaction['participants'] = self.int_participants
  
   def endinteractorRef(self, name):
       if(self.state[-2] == "participant"):
           self.cur_participant["interactor_id"] = int(self.last_nodedata)
  
   def endexperimentRef(self, name):
       self.int_experiments.append(int(self.last_nodedata))
   
   def endexperimentDescription(self, name):
       self.experiments[self.cur_experiment['id']] = self.cur_experiment
       self.cur_experiment = None

   
   def endinteraction(self, name):
       self.cur_interaction['experiments'] = self.int_experiments
       self.interactions[self.cur_interaction['id']] = self.cur_interaction
       self.cur_interaction = None
  
   def startinteractor(self, name, attrs):
       self.cur_interactor = {'id':int(attrs['id']), 'refseq_id':Missing,
                               'type_id':Missing}
       if(self.state[-2] == "participant"):
            self.cur_participant["interactor_id"] = int(attrs['id'])
   
  
   def endinteractor(self, name):
       self.interactors[self.cur_interactor['id']] = self.cur_interactor
       self.cur_interactor = None
   

   def startexperimentDescription(self, name, attrs):
       self.cur_experiment = {'id':int(attrs['id'])}
       if(self.state[-3] == "interaction"):
           self.int_experiments.append(int(attrs['id']))
        

   def startprimaryRef(self, name, attrs):
        if(attrs['id'] == 'unknown'):
            return
        if(attrs['db'] == 'psi-mi'):
            if(self.state[-3] == 'interactionDetectionMethod'):
                if(self.state[-4] == 'experimentDescription'):
                    self.cur_experiment['method_id'] = str(attrs['id'])
            elif(self.state[-3] == "interactorType"):
                self.cur_interactor['type_id'] = str(attrs['id'])
        elif(attrs['db'].lower() == 'pubmed'):
            if(self.state[-4] == 'experimentDescription' and self.state[-3] == "bibref"):
                self.cur_experiment['pubmed_id'] = str(attrs['id'])
        elif(attrs['db'] == 'refseq'):
            if(self.state[-3] == 'interactor'):
                self.cur_interactor['refseq_id'] = str(attrs['id'])

   startsecondaryRef = startprimaryRef
  
   def endshortLabel(self, name):
       if(self.state[-3] == "interactor"):
           self.cur_interactor['short_name'] = self.last_nodedata
       elif(self.state[-3] == "interactionType"):
           self.cur_interaction['type'] = self.last_nodedata
       elif(self.state[-3] == "experimentalRole"):
           self.cur_participant["role"] = self.last_nodedata
       elif(self.state[-4] == "confidence"):
           self.cur_confidence_type = self.last_nodedata
       elif(self.state[-3] == "interactionDetectionMethod"):
           self.cur_experiment['method_name'] = self.last_nodedata
   

   def endvalue(self, name):
       if(self.state[-2] == "confidence"):
           if(self.cur_confidence_type == "author-confidence"):
               self.cur_interaction["author_confidence"] = self.last_nodedata
           elif(self.cur_confidence_type == "intact confidence"):
               self.cur_interaction["intact_confidence"] = float(self.last_nodedata)
               
   def startattribute(self, name, attr):
       self.cur_attribute = attr["name"]
   
   def endattribute(self, name):
       if(self.cur_attribute == "BioGRID Evidence Code"):
           self.cur_interaction['biogrid_type'] = self.last_nodedata

   def endelse(self, name):
       pass

   def startElement(self, name, attrs):                        
       self.state.append(name)
       self.start(name, attrs)

   def endElement(self, name):
       self.end(name)
       self.state.pop()

   def characters(self, content):                 
       self.last_nodedata = content
   
   def error(self, exception):
        print "ERROR:" + str(exception)

   def warning(self, exception):
        print "WARNING: " + str(exception)

   def fatalError(self, exception):
        raise exception


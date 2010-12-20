import copy

from ..utils import util
from ..constants import *


#pylint: disable-msg=E1101
dimid = util.seqgen().next
def getNewDimid():
    return dimid()
class Dim(object):
    """Class representing a dimension."""
    __slots__ = ['id', 'name', 'shape', 'variable', 'has_missing']
    
    def __init__(self, shape, variable=0, has_missing=False, did=None, name=None):
        """Creates dimension class.

        Parameters
        ---------
        shape: shape of dimension. Can be integer >= 0 or UNDEFINED
        variable: indicates the number of parent dimension this dim depends on
        has_missing: indicates if there are Missing values
        did:   id of dimension, optional
        """
        if(not did):
            self.id = getNewDimid()
        else:
            self.id = did

        assert isinstance(self.id, int), "ID should be a integer"

        if(name is None):
            self.name = "d" + str(self.id)
        else:
            assert name.lower() == name, "Dimension name should be lower case"
            self.name = name
        
        self.has_missing = has_missing
        self.variable = variable
        self.shape = shape


    def copy(self, reid=False):
        """Returns a copy of this object"""
        res = copy.copy(self)
        if(reid):
            res.id = getNewDimid()
        return res

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self.id == other.id and self.variable == other.variable)

    def __hash__(self):
        return hash(self.id) ^ hash(self.variable)

    def __repr__(self):
        if(self.variable and (self.shape > 1 or 
                                self.shape == UNDEFINED)):
            return self.name + ":~"
        elif(self.variable):
            return self.name + ":?"
        elif(self.shape == UNDEFINED):
            return self.name + ":*"
        else:
            return self.name + ":" + str(self.shape)
            

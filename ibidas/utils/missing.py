"""
.. rubric:: Missing values
    The MissingType class implements a missing type which is used in the 
    internal representation. Types which allow missing values have to set the flag
    'has_missing' to True. 

    The only instance from MissingType that should be used is Missing (similar to None).
    This way one can test for equivalence by using 'variable_name is Missing'. 

    MissingType implements most type overloads to implement missing value behaviour. 
    E.g., Missing + 3 = Missing,   Missing & False = False, Missing | True = True,
    Missing == Missing = False, etc., similar to SQL NULL. 

    !!!To make it possible to have multiple Missing values in the same set, but use the same
    instance, __hash__ returns an incrementing value, and __eq__ returns False. 
    We have to make sure that this keeps working in Python (i.e. that set will not start to 
    use a pointer comparision shortcut). Maybe we can find a better solution in the future. 
"""
import util

hash_counter = 0
class MissingType(object):#{{{
    """MissingType value object, representing not-existing or unknown values.
       Singleton value is available in rtypes.Missing"""

    __slots__ = []

    def __new__(cls):
        if not hasattr(cls, "Missing"):
            cls.Missing = object.__new__(cls)

        return cls.Missing

    def __getnewargs__(self):
        return tuple()

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __rsub__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __mod__(self, other):
        return self

    def __rmod__(self, other):
        return self

    def __div__(self, other):
        return self

    def __rdiv__(self, other):
        return self

    def __and__(self, other):
        if(other is False or (isinstance(other,numpy.bool_) and other == False)):
            return False
        elif(isinstance(other,numpy.ndarray)):
            return other.__and__(self)
        return self

    def __rand__(self, other):
        if(other is False or (isinstance(other,numpy.bool_) and other == False)):
            return False
        elif(isinstance(other,numpy.ndarray)):
            return other.__rand__(self)
        return self

    def __or__(self, other):
        if(other is True or (isinstance(other,numpy.bool_) and other == True)):
            return True
        elif(isinstance(other,numpy.ndarray)):
            return other.__ror__(self)
        return self

    def __ror__(self, other):
        if(other is True or (isinstance(other,numpy.bool_) and other == True)):
            return True
        elif(isinstance(other,numpy.ndarray)):
            return other.__or__(self)
        return self

    def __xor__(self, other):
        return self

    def __rxor__(self, other):
        return self

    def __eq__(self, other):
        return self

    def __ne__(self, other):
        return self

    def __le__(self, other):
        return self

    def __ge__(self, other):
        return self

    def __lt__(self, other):
        return self

    def __gt__(self, other):
        return self

    def __invert__(self):
        return self

    def __pos__(self):
        return self

    def __neg__(self):
        return self

    def __abs__(self):
        return self

    def __nonzero__(self):
        return False
    
    def __hash__(self):
        global hash_counter
        hash_counter += 1
        return hash_counter

    def __repr__(self):
        return "--"

    def __call__(self, *seq, **kwds):
        return self

#Singleton value for _MissingType
Missing = MissingType()


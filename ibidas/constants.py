
UNDEFINED = -1


BCCOPY=0       #exists here, exists there
BCENSURE=1     #exists here, exists there, but has different dim identity

BCSOURCE=2     #exists here, not there

BCNEW=3        #exists not here, needs broadcasting
BCEXIST=4      #exists not here, but broadcast dim does exist


NOVAL="NOVALUE" #alternative None for functions which also accept NOne as valid input
                #but still want some params to be optional


#slice matching
COMMON_POS="COMMONPOS"
COMMON_NAME="COMMONNAME"


LASTCOMMONDIM = -1L
LCDIM = LASTCOMMONDIM

class NewDim(object):
    def __init__(self,name=None):
        self.name = name
    def __call__(self,name=None):
        return NewDim(name)
newdim = NewDim()
NEWDIM = newdim




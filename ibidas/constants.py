
UNDEFINED = -1


BCCOPY=0       #exists here, exists there
BCENSURE=1     #exists here, exists there, but has different dim identity

BCSOURCE=2     #exists here, not there

BCNEW=3        #exists not here, needs broadcasting
BCEXIST=4      #exists not here, but broadcast dim does exist


NOVAL="NOVALUE" #alternative None for functions which also accept NOne as valid input
                #but still want some params to be optional

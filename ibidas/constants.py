
UNDEFINED = -1


#data states, used in rtype to determine current 
#data form
DATA_NORMAL=0  #data in normal ibidas format
DATA_INPUT=1   #data in some input format convertable by scanner
DATA_FROZEN=2  #data in frozen state, useable for set (can be combined with DATA_NORMAL)



RS_TYPES_KNOWN = 1
RS_SLICES_KNOWN = 2
RS_ALL_KNOWN = RS_TYPES_KNOWN | RS_SLICES_KNOWN
RS_INFERRED = 8
RS_CHECK = 16

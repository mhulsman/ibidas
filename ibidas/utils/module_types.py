import util

class PlaceHolderType:
    pass

if util.check_module("SOAPpy"):
    import SOAPpy
    soap_struct = SOAPpy.Types.structType
else:
    soap_struct = PlaceHolderType

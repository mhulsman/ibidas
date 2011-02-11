class Expression(object):
    __slots__ = ["etype","eobj","in_slices","out_slices","all_slices"]
    def __init__(self, etype, eobj):
        self.etype = etype
        self.eobj = eobj
        self.in_slices = set()
        self.out_slices = set()
        self.all_slices = set()

    def addInSlice(self, in_slice):
        self.in_slices.add(in_slice)

    def addOutSlice(self, out_slice):
        self.out_slices.add(out_slice)

    def addAllSlice(self, between_slice):
        self.all_slices.add(between_slice)


    def __str__(self):
        return str(self.etype)

class BinFuncElemExpression(Expression):
    pass 

class FilterExpression(Expression):
    pass



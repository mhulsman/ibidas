class Infix:
    def __init__(self, function):
        self.function = function
    def __lt__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __gt__(self, other):
        return self.function(other)
    def __rlshift__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x)) 
    def __rshift__(self, other):
        return self.function(other)
    def __call__(self, value1, value2):
        return self.function(value1, value2)

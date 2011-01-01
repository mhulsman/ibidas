import unittest
import numpy

from ibidas.utils import nested_array
from ibidas.utils import cutils
from ibidas.itypes import detector
from ibidas import *

class TestNestedArray(unittest.TestCase):
    def setUp(self):
        self.scalar = 1
        self.array = numpy.array([1,2,3,4,5])
        self.matrix = numpy.array([[1,2,3,4,5],[1,2,3,4,5]])
        self.nestedarray = cutils.darray([[1,2,3],[1,2],[1]])
        self.nestedmatrix = cutils.darray([[[1,2],[2,4],[3,5]],[[1,3],[5,2]],[[1,2]]])
        self.nestednestedarray = cutils.darray([[[1,2],[2,4,5],[]],[[1],[2]],[[1,4,5]]])
        self.nestednestedmatrix = cutils.darray([[[[1,2],[3,4]],[[5,6],[7,9],[9,8]],[[]]],[[[1,4]],[[2,5]]],[[[1,5],[4,7],[8,5]]]])
        self.arraynestednestedmatrix = cutils.darray([self.nestednestedmatrix, self.nestednestedmatrix, self.nestednestedmatrix])

        self.data = [self.scalar, self.array, self.matrix, self.nestedarray, self.nestedmatrix, self.nestednestedarray, self.nestednestedmatrix, self.arraynestednestedmatrix]
        
    def test_init(self):
        k = [rep(elem)() for elem in self.data]


    def test_same(self):
        orig = [rep(elem) for elem in self.data]
        reload = [rep(elem()) for elem in orig]
        [left == right for left,right in zip(orig,reload)]


        
if __name__ == "__main__":
    unittest.main()

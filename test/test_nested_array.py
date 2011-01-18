import unittest
import numpy

from ibidas.utils import nested_array
from ibidas.utils import cutils
from ibidas.itypes import detector
from ibidas import *


class TestData(unittest.TestCase):
    def setUp(self):
        self.data = None

    def test_init(self):
        k = rep(self.data)()

    def test_same(self):
        orig = rep(self.data) 
        reload = rep(orig()) 
        self.assertTrue(orig ==+ reload)
    

class TestScalar(TestData):
    def setUp(self):
        self.data = 1

class TestArray(TestData):
    def setUp(self):
        self.data = numpy.array([1,2,3,4,5])
    
    def test_type(self):
        k = rep(self.data)()
        self.assertTrue(k.dtype == self.data.dtype)

    def test_filter(self):
        k = rep(self.data)
        self.assertTrue(k[k == 3]==3)
        self.assertFalse(k[k == 3] == 4)

class TestMatrix(TestArray):
    def setUp(self):
        self.data = numpy.array([[1,2,3,4,5],[1,2,3,4,5]])

class TestNestedArray(TestArray):
    def setUp(self):
        self.data = cutils.darray([[1,2,3],[1,2],[1]])

class TestNestedMatrix(TestArray):
    def setUp(self):
        self.data = cutils.darray([[[1,2],[2,4],[3,5]],[[1,3],[5,2]],[[1,2]]])

class TestNestedMatrixString(TestArray):
    def setUp(self):
        self.data = cutils.darray([[["abc","abcd"],["def","defg"],["abg","fev"]],[["zeK","sdf"],["sdf","sdfff"]],[["sdf","kjl"]]])
    
    def test_filter(self):
        k = rep(self.data)
        self.assertTrue(k[k == "def"]=="def")
        self.assertFalse(k[k == "def"] == "defg")

class TestNestedNestedArray(TestArray):
    def setUp(self):
        self.data = cutils.darray([[[1,2],[2,3,3,3,4,5],[]],[[1],[2]],[[1,4,5]]])

class TestNestedNestedMatrix(TestArray):
    def setUp(self):
        self.data = cutils.darray([[[[1,2],[3,4]],[[5,6],[7,9],[9,8]],[[]]],[[[1,4]],[[2,5]]],[[[1,5],[4,7],[8,5]]]])

class TestArrayNestedNestedMatrix(TestArray):
    def setUp(self):
        nnmatrix = cutils.darray([[[[1,2],[3,4]],[[5,6],[7,9],[9,8]],[[]]],[[[1,4]],[[2,5]]],[[[1,5],[4,7],[8,5]]]])
        self.data = cutils.darray([nnmatrix, nnmatrix, nnmatrix])

        
if __name__ == "__main__":
    unittest.main()

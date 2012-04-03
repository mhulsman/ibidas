import unittest
import numpy

from ibidas.utils import nested_array
from ibidas.utils import cutils,util
from ibidas.itypes import detector
from ibidas import *


class TestData(unittest.TestCase):
    def setUp(self):
        self.data = None

    def test_init(self):
        k = Rep(self.data)()

    def test_same(self):
        orig = Rep(self.data) 
        reload = Rep(orig()) 
        self.assertTrue(orig ==+ reload)
    
    def getRep(self):
        if(hasattr(self,'dtype')):
            return Rep(self.data,dtype=self.dtype[0])
        else:
            return Rep(self.data)
    
    def test_type(self):
        if(hasattr(self,'dtype')):
            for elem in self.dtype:
                self.assertTrue(rep(self.data,dtype=elem) ==+ Rep(self.data))

class TestScalar(TestData):
    def setUp(self):
        self.data = 1

class TestArray(TestData):
    def setUp(self):
        self.data = numpy.array([1,2,3,4,5])
        self.fres0 = 1
    
    def test_type(self):
        k = Rep(self.data)()
        self.assertTrue(k.dtype == self.data.dtype)

    def test_filter(self):
        k = Rep(self.data)
        self.assertTrue(k[k == 3]==3)
        self.assertFalse(k[k == 3] == 4)

    def test_filter2(self):
        k = Rep(self.data)
        self.assertTrue(k[0] == self.fres0)



class TestMatrix(TestArray):
    def setUp(self):
        self.data = numpy.array([[1,2,3,4,5],[1,2,3,4,5]])
        self.fres0 = numpy.array([1,1])
        self.transpose_couples = [((1,0),(1,0))]

    def test_tranpose(self):
        r = self.getRep()
        pidx = range(r.Depth)
        self.assertTrue(r.Transpose(pidx) ==+ r)
       
        if(hasattr(self,'transpose_couples')):
           for i in range(len(self.transpose_couples)):
               t1,t2 = self.transpose_couples[i]
               self.assertTrue(r.Transpose(t1).Transpose(t2) ==+ r)
        
    def test_flat(self):
       f1 = Rep(self.data)
       #backward
       while(f1.Depth >= 2):
            f1 = f1.Flat(None)
       #forward
       f2 = Rep(self.data)
       while(f2.Depth >= 2):
            f2 = f2.Flat(f2._slices[0].dims[1].name)
       #all
       f3 = Rep(self.data).FlatAll()
       #compare
       self.assertTrue(f1 ==+ f2)
       self.assertTrue(f2 ==+ f3)
       self.assertFalse(f2 ==+ (f3.Sort()))


    def test_filter3(self):
        k = Rep(self.data)
        self.assertTrue(k[:1].Flat() ==+ k[0])

    def test_filter4(self):
        k = Rep(self.data)
        self.assertTrue(k[Rep([0],unpack=False)].Flat() ==+ k[0])
       

class TestNestedArray(TestMatrix):
    def setUp(self):
        self.data = util.darray([[1,2,3],[1,2],[1]])
        self.fres0 = util.darray([1,1,1])

class TestNestedMatrix(TestMatrix):
    def setUp(self):
        self.data = util.darray([[[1,2],[2,4],[3,5]],[[1,3],[5,2]],[[1,2]]])
        self.fres0 = util.darray([[1,2,3],[1,5],[1]])
        self.transpose_couples = [((0,2,1),(0,2,1)),((2,0,1),(1,2,0))]

class TestNestedMatrixString(TestMatrix):
    def setUp(self):
        self.data = util.darray([[["abc","abcd"],["def","defg"],["abg","fev"]],[["zeK","sdf"],["sdf","sdfff"]],[["sdf","kjl"]]])
        self.fres0 = util.darray([["abc","def","abg"],["zeK","sdf"],["sdf"]])
        self.transpose_couples = [((0,2,1),(0,2,1)),((2,0,1),(1,2,0))]
    
    def test_filter(self):
        k = Rep(self.data)
        self.assertTrue(k[k == "def"]=="def")
        self.assertFalse(k[k == "def"] == "defg")

class TestNestedNestedArray(TestMatrix):
    def setUp(self):
        self.data = util.darray([[[1,2],[2,3,3,3,4,5],[]],[[1],[2]],[[1,4,5]]])
        self.fres0 = util.darray([[1,2,[]],[1,2],[1]])

    def test_filter2(self):
        k = Rep(self.data)
        self.assertRaises(Exception,k[0])

    def test_filter3(self):
        pass 

    def test_filter4(self):
        pass

class TestNestedNestedMatrix(TestMatrix):
    def setUp(self):
        self.data = util.darray([[[[1,2],[3,4]],[[5,6],[7,9],[9,8]],[[4,9]]],[[[1,4]],[[2,5]]],[[[1,5],[4,7],[8,5]]]])
        self.transpose_couples = [((0,1,3,2),(0,1,3,2)),((0,3,1,2),(0,2,3,1)),((3,0,1,2),(1,2,3,0))]

    def test_filter2(self):
        k = Rep(self.data)
        self.assertRaises(Exception,k[3])

class TestArrayNestedNestedMatrix(TestMatrix):
    def setUp(self):
        nnmatrix = util.darray([[[[1,2],[3,4]],[[5,6],[7,9],[9,8]],[[8,9]]],[[[1,4]],[[2,5]]],[[[1,5],[4,7],[8,5]]]])
        self.data = util.darray([nnmatrix, nnmatrix, nnmatrix])
        self.transpose_couples = [((1,2,3,4,0),(4,0,1,2,3)),((1,0,2,3,4),(1,0,2,3,4)),((1,2,0,3,4),(2,0,1,3,4)),((1,2,3,0,4),(3,0,1,2,4))]

    def test_filter2(self):
        k = Rep(self.data)
        self.assertRaises(Exception,k[3])

class TestNestedVarMatrix(TestMatrix):
    def setUp(self):
        self.data = util.darray([[[1,2],[3,4],[5,6]],[[4,5,6],[7,8,9]]])
        self.dtype=["[arr]<[var1:.]<[var2:*.]<int64"]
        self.fres0 = util.darray([[1,3,5],[4,7]])
        self.transpose_couples = [((0,2,1),(0,2,1))]
        
if __name__ == "__main__":
    unittest.main()

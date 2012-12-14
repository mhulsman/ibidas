import unittest
import numpy

from ibidas.utils import util
from ibidas import *


class TestBugs(unittest.TestCase):
    def test_posbroadcast_match(self):
        yeastract = Get.yeast.yeastract()
        yeastract = yeastract.GroupBy(_.trans_factor)[:10].Copy()
        str(yeastract |Match(_.target)| yeastract[:,newdim])
        str(yeastract |Match(_.target,mode="pos")| yeastract[:,newdim])

    
    def test_filterafter_complexmatch(self):
        yeastract = Get.yeast.yeastract()
        yeastract = yeastract.GroupBy(_.trans_factor)[:10].Copy()
        res = yeastract |Match(_.target,mode="pos")| yeastract[:,newdim]
   
        str(res[0,0,0])
        str(res[:,:,0])
        self.assertTrue(all(All((res[0,...] ==+ res[0,:,:]).FlatAll())()))
        str(res[:2,...])
        str(res[[0,1],...])
        str(res[Rep([0,1]),...])
        str(res[_.target == "YGR258c"])

    def test_posbroadcast(self):
        x = Rep([1,2,3])
        self.assertTrue((x ++ x).Depth == 1)
        self.assertTrue((x ++ x[:,newdim]).Depth == 2)
        self.assertTrue((x ++ x[...,newdim]).Depth == 2)

    def test_filter(self):
        yeastract = Get.yeast.yeastract()
        y = yeastract.GroupBy(_.trans_factor)[:10].Copy()
        str(y.Get(_.trans_factor, _.target[:,newdim][:10]).Count())
    
    def test_slicesunknownmatch(self):
        x = Rep(['test']).Each(str.upper)
        str(x |Match(_.Each(str.upper))| x)


    def test_filter_on_transposed_nested(self):
        z = Rep((['a','b','b','c'],[[1,2,3,4],[1,2,3,4],[4,3,2,1]]))
        res = z.To(_.f1, Do=_.Transpose()).GroupBy(_.f0).To(_.f1, Do=_.Mean(dim=1)).To(_.f1, Do=_.Transpose()).Filter(slice(0,5),dim=1)
        str(res)

    def test_add_on_matrix_plus_nested(self):
        data = [([6,2],[0.5]), ([3,4], [0.3,0.4]), ([6,4], [0.8,0.2])]
        z = Rep(data)
        str(z.f0 + z.f1)
        str(z.f1 + z.f0)
        str(z.f0.Transpose() + z.f1)
        str(z.f1 + z.f0.Transpose())

    def test_add_nothing(self):
        data = [(163818, 159108, 455, [[  9, 455]]), (181601, 159108, 748, [[  2, 559],  [  7, 212],  [ 13, 364]])];
        z    = Rep(data);
        db   = Get.in_memory_db();
        db.Store('t', z[:0]);


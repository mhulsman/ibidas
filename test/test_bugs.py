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

        str(res[2])
        str(res[:2])
        str(res[Rep([2,3],unpack=False)])
        str(res[_.target == "YGR258c"])
        str(res[[2,3]])

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

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

    def test_posbroadcast(self):
        x = Rep([1,2,3])
        self.assertTrue((x ++ x).Depth == 1)
        self.assertTrue((x ++ x[:,newdim]).Depth == 2)
        self.assertTrue((x ++ x[...,newdim]).Depth == 2)

    def test_filter(self):
        yeastract = Get.yeast.yeastract()
        y = yeastract.GroupBy(_.trans_factor)[:10].Copy()
        str(y.Get(_.trans_factor, _.target[:,newdim][:10]).Count())
       

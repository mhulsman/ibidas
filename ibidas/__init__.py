"""
The ibidas module contains all main functions for working with ibidas objects.
"""

__all__ = ["rep","_",
           "rarray","rlist","rtuple","combine","harray",
           "pos","argsort",
           "rany","rall",
           "rmax","rmin",
           "argmin","argmax",
           "mean","median",
           "rsum",
           "count",
           "bcast","createType",
           "newdim","Missing"]

from utils import delay_import
from utils.context import _
from utils.missing import Missing
from utils.infix import Infix,RevInfix
from itypes import *
from wrappers.wrapper_py import rep
from representor import newdim
from repops_dim import rlist, rarray
from repops_multi import Broadcast as bcast, Combine as combine, sort
from repops_slice import RTuple as rtuple, HArray as harray
from repops_funcs import ArgSort as argsort, Pos as pos, Any as rany, All as rall,\
                         Max as rmax, Min as rmin, ArgMax as argmax, ArgMin as argmin,\
                         Sum as rsum, Mean as mean, Median as median, Count as count

within = Infix(repops_funcs.Within)
contains = RevInfix(repops_funcs.Within)

delay_import.perform_delayed_imports()


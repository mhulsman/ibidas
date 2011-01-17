"""
The ibidas module contains all main functions for working with ibidas objects.
"""

__all__ = ["rep","_",
           "rarray","rlist","rtuple","combine","harray",
           "pos","argsort",
           "rany","rall",
           "bcast","createType",
           "newdim","Missing"]

from utils import delay_import
from utils.context import _
from utils.missing import Missing
from itypes import *
from wrappers.wrapper_py import rep
from representor import newdim
from repops_dim import rlist, rarray
from repops_multi import Broadcast as bcast, Combine as combine
from repops_slice import RTuple as rtuple, HArray as harray
from repops_funcs import ArgSort as argsort, Pos as pos, Any as rany, All as rall

delay_import.perform_delayed_imports()


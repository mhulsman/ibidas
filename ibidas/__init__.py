"""
The ibidas module contains all main functions for working with ibidas objects.
"""

__all__ = ["Rep","Read","Connect","_",
           "Array","Tuple","Combine","HArray",
           "Stack",
           "Pos","Argsort",
           "Any","All",
           "Max","Min",
           "Argmin","Argmax",
           "Mean","Median",
           "Sum",
           "Count",
           "Broadcast","CreateType",
           "newdim","Missing",
           "Corr","In","Contains",
           "Fetch","Serve","Get",
           "Load","Save",'newdim',
           "Invert","Abs", "Negative",
           ]

from utils import delay_import
from utils.util import save_rep as Save, load_rep as Load
from utils.context import _
from utils.missing import Missing
from utils.infix import Infix,RevInfix
from itypes import createType as CreateType
from wrappers.wrapper_py import Rep
from wrappers.wrapper_tsv import TSVRepresentor
from wrappers.wrapper_sql import open_db
from representor import newdim
import repops_dim 
from repops_multi import Broadcast, Combine, Sort, Stack
import repops_slice 
import repops_funcs
from download_cache import DownloadCache
from pre import predefined_sources as Get
from server import Serve
from constants import *

Fetch = DownloadCache()
In = Infix(repops_funcs.Within)
Contains = Infix(repops_funcs.Contains)

Pos = repops.delayable(default_params="#")(repops_funcs.Pos)
Argsort = repops.delayable()(repops_funcs.Argsort)
Argmax = repops.delayable()(repops_funcs.Argmin)
Argmin = repops.delayable()(repops_funcs.Argmax)
Sum = repops.delayable()(repops_funcs.Sum)
Any = repops.delayable()(repops_funcs.Any)
All = repops.delayable()(repops_funcs.All)
Max = repops.delayable()(repops_funcs.Max)
Min = repops.delayable()(repops_funcs.Min)
Mean = repops.delayable()(repops_funcs.Mean)
Median = repops.delayable()(repops_funcs.Median)
Count = repops.delayable()(repops_funcs.Count)
Corr = repops.delayable()(repops_funcs.Corr)
Invert = repops.delayable()(repops_funcs.Invert)
Abs = repops.delayable()(repops_funcs.Abs)
Negative = repops.delayable()(repops_funcs.Negative)


HArray = repops.delayable(nsources=UNDEFINED)(repops_slice.HArray)
Tuple = repops.delayable(nsources=UNDEFINED)(repops_slice.Tuple)
Array = repops.delayable()(repops_dim.Array)

def Read(url, **kwargs):
    format = kwargs.pop('format','tsv')

    if(format == 'tsv'):
        return TSVRepresentor(url, **kwargs) 
    else:
        raise RuntimeError("Unknown format specified")

def Connect(url, **kwargs):
    format = kwargs.pop('format','db')
    if(format == "db"):
        return open_db(url, **kwargs)
    else:
        raise RuntimeError("Unknown format specified")

delay_import.perform_delayed_imports()

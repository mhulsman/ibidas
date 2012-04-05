"""
The ibidas module contains all main functions for working with ibidas objects.
"""

__all__ = ["Rep","Read","Connect","_","CyNetwork",'Unpack',
           "Array","Tuple","Combine","HArray",
           "Stack","Intersect","Union","Except","Difference",
           "Pos","Argsort","Rank","IsMissing","CumSum",
           "Any","All",
           "Max","Min",
           "Argmin","Argmax",
           "Mean","Median",
           "Sum",
           "Count","Match",
           "Broadcast","CreateType","MatchType",
           "newdim","Missing",
           "Corr","In","Contains",
           "Fetch","Serve","Get","Alg",
           "Load","Save",'newdim',
           "Invert","Abs", "Negative",
           ]

from utils import delay_import
from utils.util import save_rep, load_rep, save_csv
from utils.context import _
from utils.missing import Missing
from utils.infix import Infix
from itypes import createType as CreateType, matchType as MatchType
from wrappers.python import Rep
from representor import newdim
import repops_dim 
from repops_multi import Broadcast, Combine, Sort
import repops_slice 
import repops_funcs
from download_cache import DownloadCache, Unpack
from pre import predefined_sources as Get
from algs import predefined_algs as Alg
from wrappers.cytoscape import CyNetwork
from server import Serve
from constants import *

Fetch = DownloadCache()
In = Infix(repops_funcs.Within)
Contains = Infix(repops_funcs.Contains)
Match = Infix(repops_multi.Match)
Stack = Infix(repops_multi.Stack)
Intersect = Infix(repops_multi.Intersect)
Union = Infix(repops_multi.Union)
Except = Infix(repops_multi.Except)
Difference = Infix(repops_multi.Difference)

Pos = repops.delayable(default_params="#")(repops_funcs.Pos)
IsMissing = repops.delayable()(repops_funcs.IsMissing)
Argsort = repops.delayable()(repops_funcs.Argsort)
Rank = repops.delayable()(repops_funcs.Rank)
CumSum = repops.delayable()(repops_funcs.CumSum)
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
        from wrappers.tsv import TSVRepresentor
        return TSVRepresentor(url, **kwargs) 
    if(format == 'matrix_tsv'):
        from wrappers.matrix_tsv import MatrixTSVRepresentor
        return MatrixTSVRepresentor(url, **kwargs) 
    elif(format == 'xml'):
        from wrappers.xml_wrapper import XMLRepresentor
        return XMLRepresentor(url, **kwargs) 
    elif(format == 'psimi'):
        from wrappers.psimi import read_psimi
        return read_psimi(url, **kwargs)
    else:
        raise RuntimeError("Unknown format specified")

def Connect(url, **kwargs):
    format = kwargs.pop('format','db')
    if(format == "db"):
        from wrappers.sql import open_db
        return open_db(url, **kwargs)
    else:
        raise RuntimeError("Unknown format specified")


def Save(r, filename):
    if filename.endswith('tsv') or filename.endswith('csv') or filename.endswith('tab'):
        save_csv(r, filename);
    else:
        save_rep(r, filename);


def Load(filename,**kwargs):
    if filename.endswith('tsv') or filename.endswith('csv') or filename.endswith('tab'):
        from wrappers.tsv import TSVRepresentor
        return TSVRepresentor(filename, **kwargs) 
    else:
        return load_rep(filename)

delay_import.perform_delayed_imports()






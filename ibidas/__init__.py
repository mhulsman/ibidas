"""
The ibidas module contains all main functions for working with ibidas objects.
"""

__all__ = ["Rep","Read","Import","Connect","_","CyNetwork",'Unpack', "Addformat",
           "Array","Tuple","Combine","HArray",
           "Stack","Intersect","Union","Except","Difference",
           "Pos","Argsort","Rank","IsMissing","CumSum",
           "Any","All",
           "Max","Min",
           "Argmin","Argmax",
           "Mean","Median",
           "Sum",
           "Count","Match","Join",
           "Broadcast","CreateType","MatchType",
           "newdim","NEWDIM","LCDIM","LASTCOMMONDIM","COMMON_NAME",'COMMON_POS', "Missing",
           "Corr","In","Contains",
           "Fetch","Serve","Get","Alg",
           "Load","Save",
           "Invert","Abs", "Negative",
           "Add","Subtract","Multiply","Modulo","Divide","FloorDivide","And","Or","Xor","Power","Equal","NotEqual","LessEqual","Less","GreaterEqual","Greater","Each"
           ]

from utils import delay_import
from utils.util import save_rep, load_rep, save_csv
from utils.context import _
from utils.missing import Missing
from utils.infix import Infix
from itypes import createType as CreateType, matchType as MatchType
from wrappers.python import Rep
from constants import *
from repops import Detect
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
Join = Infix(repops_multi.Join)
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


Add = repops.delayable()(repops_funcs.Add)
Subtract = repops.delayable()(repops_funcs.Subtract)
Multiply = repops.delayable()(repops_funcs.Multiply)
Modulo= repops.delayable()(repops_funcs.Modulo)
Divide = repops.delayable()(repops_funcs.Divide)
FloorDivide = repops.delayable()(repops_funcs.FloorDivide)
And = repops.delayable()(repops_funcs.And)
Or = repops.delayable()(repops_funcs.Or)
Xor = repops.delayable()(repops_funcs.Xor)
Power = repops.delayable()(repops_funcs.Power)
Equal = repops.delayable()(repops_funcs.Equal)
NotEqual = repops.delayable()(repops_funcs.NotEqual)
LessEqual = repops.delayable()(repops_funcs.LessEqual)
Less = repops.delayable()(repops_funcs.Less)
GreaterEqual = repops.delayable()(repops_funcs.GreaterEqual)
Greater = repops.delayable()(repops_funcs.Greater)
Each = repops.delayable()(repops_funcs.Each)


HArray = repops.delayable(nsources=UNDEFINED)(repops_slice.HArray)
Tuple = repops.delayable(nsources=UNDEFINED)(repops_slice.Tuple)
Array = repops.delayable()(repops_dim.Array)



def fimport_tsv(url, **kwargs):
    from wrappers.tsv import TSVRepresentor
    return TSVRepresentor(url, **kwargs) 

def fimport_matrixtsv(url, **kwargs):
    from wrappers.matrix_tsv import MatrixTSVRepresentor
    return MatrixTSVRepresentor(url, **kwargs)

def fimport_xml(url, **kwargs):
    from wrappers.xml_wrapper import XMLRepresentor
    return XMLRepresentor(url, **kwargs) 

def fimport_psimi(url, **kwargs):
    from wrappers.psimi import read_psimi
    return read_psimi(url, **kwargs)

def fimport_fasta(url, **kwargs):
    from wrappers.fasta import read_fasta;
    return read_fasta(url, **kwargs);


formats = { 'tsv' : fimport_tsv, 'csv' : fimport_tsv,
            'tsv_matrix' : fimport_matrixtsv,
	    'xml' : fimport_xml,
	    'psimi' : fimport_psimi,
	    'fasta' : fimport_fasta, 'fa' : fimport_fasta, 'fas' : fimport_fasta
	  };

def Addformat(ext, read_fn):
    formats[ext] = read_fn;

def Import(url, **kwargs):

  from os.path import splitext;
  detect=kwargs.pop('detect', True);

  base = url;

  while True:
    (base, ext) = splitext(base);
    ext = ext.split('.')[1] if ext else 'tsv';
    format = kwargs.pop('format', ext).lower();

    if not format:
        raise RuntimeError("Unknown format specified")
    if format not in formats:
        continue;
    else:
        data = formats[format](url, **kwargs);
        return data.Detect() if detect else data;

Read = Import

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






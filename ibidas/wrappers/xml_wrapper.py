from xml.dom.minidom import parse
import os

import wrapper
from ..itypes import rtypes
from .. import repops_slice, ops
from ..constants import *

import python
from ..itypes import detector
from ..utils import nested_array, util

class XMLRepresentor(wrapper.SourceRepresentor):
    def __init__(self, filename):
        dom = parse(filename)
        allowed_nodes = set([dom.ELEMENT_NODE])
        res = parse_node(dom, allowed_nodes)
        if isinstance(res,list) and len(res) == 1:
            res = res[0]

        det = detector.Detector()
        det.process(res)
        dtype = det.getType()

        slice = XMLOp(res, dtype, "data")
        if(slice.type.__class__ is rtypes.TypeRecordDict):
            nslices = repops_slice.UnpackTuple._apply(slice)
        else:
            nslices = [slice]
        self._initialize(tuple(nslices))


class XMLOp(ops.ExtendOp):
    __slots__ = ["data"]

    def __init__(self, data, rtype, name):
        self.data = data
        ops.ExtendOp.__init__(self,name=name,rtype=rtype)

    def py_exec(self):
        ndata = nested_array.NestedArray(self.data,self.type)
        return python.ResultOp.from_slice(ndata,self)


def parse_node(node, allowed_nodes):
    if node.nodeType in allowed_nodes:
        res = {'node_name':node.nodeName}
        if node.hasChildNodes():
            res['node_children'] = [parse_node(n,allowed_nodes) for n in node.childNodes if n.nodeType in allowed_nodes]
        if node.hasAttributes():
            res.update(dict(node.attributes.items()))
        if not node.nodeValue is None and not (isinstance(node.nodeValue,str) and node.nodeValue.strip() == ""):
            res['node_value'] = node.nodeValue
        return res 
    else:
        return [parse_node(n, allowed_nodes) for n in node.childNodes if n.nodeType in allowed_nodes]

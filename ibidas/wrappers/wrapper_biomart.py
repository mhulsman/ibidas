import urllib
import xml.dom.minidom
import csv

import wrapper
_delay_import_(globals(),"wrapper_py","Result")
_delay_import_(globals(),"..itypes","rtypes","dimensions")
_delay_import_(globals(),"slices")

def connect_biomart(url="http://www.biomart.org/biomart/martservice", query=None):
    if(query is None):
        return BiomartConnection(url)
    else:
        return BiomartRepresentor(url, query)
    return Connection(con)

class BiomartConnection(object):
    def __init__(self, engine):
        pass    

    def __getattr__(self, name):
        return self.tabledict[name]

    def _getAttributeNames(self):
        return self.tabledict.keys()
    
    def __repr__(self):
        res = "Database: " + str(self.engine.url) + "\n"
        tablenames = self.tabledict.keys()
        tablenames.sort()
        res += "Tables: " + ", ".join([str(tname) for tname in tablenames])
        return res

class BiomartMart(object):
    pass

class BiomartDataset(object):
    pass

class BiomartDatasetSection(object):
    pass

class BiomartAttribute(object):
    pass

class BiomartRepresentor(wrapper.SourceRepresentor):
    _select_indicator = None
    _select_executor = None
    
    def __init__(self, url, query):
        querydom = xml.dom.minidom.parseString(query)
        names = [str(node.getAttribute("name")) for node in 
                            querydom.getElementsByTagName("Attribute")]
        
        dataset_name = str(querydom.getElementsByTagName("Dataset")[0].getAttribute("name"))
        ndims = (dimensions.Dim("*", name=dataset_name.lower()),)


        subtypes = (rtypes.unknown,) * len(names)
        ntype = rtypes.TypeTuple(subtypes=subtypes, fieldnames=names)
        ntype = rtypes.TypeArray(subtypes=(ntype,), dims=ndims)
        tslices = (slices.Slice(dataset_name, ntype),)

        query_node = querydom.getElementsByTagName("Query")[0]
        query_node.setAttribute("header", "0")
        query_node.setAttribute("count", "0")
        assert query_node.getAttribute("formatter") == "TSV", \
                    "Only TSV format supported currently"


        self._encoded_query = url + "?" + urllib.urlencode([('query', str(querydom.toxml()))])


        all_slices = dict([(slice.id, slice) for slice in tslices])
        wrapper.SourceRepresentor.__init__(self, all_slices, tslices)

    def pyexec(self, executor):
        req = urllib.urlopen(self._encoded_query)
        reader = csv.reader(req, delimiter="\t")
        res = {}
        res[self._active_slices[0].id] = [row for row in reader]
        return Result(res)



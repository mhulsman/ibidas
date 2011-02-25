from ibidas.utils import util
_delay_import_(globals(),"ibidas","*")

class Pre(object):
    def register(self, func, name=None, category=None):
        if category:
            if not category in self.__dict__:
                self.__dict__[category] = Pre()
            self.__dict__[category].register(func, name)
        else:
            if(name is None):
                name = func.__name__
            setattr(self,name,func)
predefined_sources = Pre()


def yeast_feats():
    """Returns table of yeast genome features,
       from:  http://downloads.yeastgenome.org/chromosomal_feature/SGD_features.tab
    """

    rtype = """[feats:*]<(sgdid=bytes, feat_type=bytes, feat_qual=bytes, feat_name=bytes, gene_name=bytes, 
                          gene_aliases=bytes, feat_parent_name=bytes, sgdid_alias=bytes, chromosome=bytes, 
                          start=bytes, stop=bytes, strand=bytes[1], genetic_pos=bytes, coordinate_version=bytes[10], 
                          sequence_version=bytes, description=bytes)"""
    
    res = Read(Fetch("http://downloads.yeastgenome.org/chromosomal_feature/SGD_features.tab"),dtype=rtype)

    splitfunc = lambda x: x.split("|")
    res = res.To(_.gene_aliases,  Do=_.Each(splitfunc, dtype="[aliases:~]<bytes").Elem()[_ != ""])
    res = res.To(_.start, _.stop, Do=_.Cast("int$"))
    res = res.To(_.genetic_pos,   Do=_.Cast("real64$"))
    return res.Copy()
predefined_sources.register(yeast_feats,name="genomic_feats",category="yeast")  


def yeast_aliases(feats):
    """Given a yeast feats table (see ``yeast_feats``), returns a 
       two column table with the official feature names and a column with aliases.

       Can be used to map names to official names (with match)"""

    res = Stack(feats.Get(_.feat_name, _.gene_aliases).Flat(), 
                feats.Get(_.feat_name, _.gene_name), 
                feats.Get(_.feat_name, _.feat_name))
    res = res/("feat_name", "alias")
    res = res%"feat_aliases"
    res = res[_.alias != ""]
    return res.Unique().Copy()
predefined_sources.register(yeast_aliases,name="name_aliases",category="yeast")  


def yeast_phosphosites():
    url = "http://www.phosphogrid.org/downloads/phosphosites.txt"
    rtype = """[phosphosites:*]<(orf_name=bytes, gene_name=bytes, phospho_site=bytes, site_evidence=bytes, 
               site_evidence_pubmed=bytes, site_conditions=bytes, site_conditions_pubmed=bytes, site_functions=bytes, site_functions_pubmed=bytes, 
               kinases_orfs=bytes, kinases_genes=bytes, kinases_evidence=bytes, kinases_evidence_pubmed=bytes, phosphatases_orfs=bytes, 
               phosphatases_genes=bytes, phosphatases_evidence=bytes, phosphatases_evidence_pubmed=bytes, sequence=bytes, notes=bytes)"""
    res = Read(Fetch(url),fieldnames=True,dtype=rtype)
    splitfunc = lambda x: x.split("|")
    res = res.To(_.site_evidence,  Do=_.Each(splitfunc, dtype="[site_evidences:~]<bytes").Elem()[_ != "-"])
    res = res.To(_.site_evidence_pubmed,  Do=_.Each(splitfunc, dtype="[site_evidence_pubmeds:~]<bytes").Elem()[_ != "-"])
    res = res.To(_.kinases_orfs,  Do=_.Each(splitfunc, dtype="[kinases:~]<bytes").Elem()[_ != "-"])
    res = res.To(_.phosphatases_orfs,  Do=_.Each(splitfunc, dtype="[phosphatases:~]<bytes").Elem()[_ != "-"])
    return res.Copy()
predefined_sources.register(yeast_phosphosites,name="phosphosites",category="yeast")  

def in_memory_db():
    """Returns an empty in memory database"""
    return Connect("sqlite:///:memory:");
predefined_sources.register(in_memory_db)


def yeastract(url="http://www.yeastract.com/download/RegulationTwoColumnTable_Documented_20101213.tsv.gz"):
    """Downloads documented transcription factor regulation interactions from yeastract"""
    rtype = "[tftargets:*]<(trans_factor=bytes, target=bytes)"
    res = Read(Fetch(url),dtype=rtype)
    return res.Copy()
predefined_sources.register(yeastract,category="yeast")


def string_interactions(dburl, species="Saccharomyces cerevisiae"):
    """Given a Postgres db with String data, specified in dburl, and a species, returns all interactions and their score.

    The database data can be obtained from String.

    example url: "postgres://username:password@hostname:port/string_dbname"

    Use ``connect`` to access the whole database::
       
       #Get available species names: 
       >>> connect(dburl).items.species.offical_name

    """
    z = Connect(dburl)
    inter  = z.items.species |Match| z.network.protein_protein_links
    inter  = inter[_.official_name == species]
    inter  = inter |Match(_.protein_id_a, _.protein_id)| z.items.proteins//"left"
    inter  = inter |Match(_.protein_id_b, _.protein_id)| z.items.proteins//"right"
    return   inter.Get(_.left.preferred_name/"left", 
                       _.right.preferred_name/"right", 
                       _.combined_score) % "interactions"
predefined_sources.register(string_interactions)


def omim_genemap():
    """The omim genemap data"""
    rtype = """[omim:*]<(chr_map_entry_nr=bytes, month=bytes, day=bytes, year=bytes, location=bytes, gene_names=bytes, 
                       gene_status=bytes[1], title=bytes, f8=bytes, mim=bytes, method=bytes, comments=bytes, f12=bytes, 
                       disease1=bytes, disease2=bytes, disease3=bytes, mouse=bytes, reference=bytes)"""
    
    res = Read(Fetch("ftp://ftp.ncbi.nih.gov/repository/OMIM/genemap"),dtype=rtype)
    
    res = res.Get(HArray(_.disease1, _.disease2, _.disease3)[_ != " "]/"disease", "~")
    splitfunc = lambda x: x.split(", ")
    res = res.To(_.gene_names,  Do=_.Each(splitfunc, dtype="[symbols:~]<string").Elem()[_ != ""])
    res = res.To(_.method,      Do=_.Each(splitfunc, dtype="[methods:~]<string").Elem()[_ != ""])
    res = res.To(_.disease,     Do=_.Sum().Each(omim_disease_parse,"[diseases:~]<string").Elem()[_ != ""])
    res = res.To(_.day, _.month, _.year, Do=_.Cast("int$"))
    res = res.To(_.mim, Do=_.Cast("int"))

    return res.Without(_.f8, _.f12, _.disease1, _.disease2, _.disease3).Copy()
predefined_sources.register(omim_genemap, category="human")

def omim_disease_parse(x):
    x = x.replace('{','').replace('}','')
    elems =  x.split('; ')
    return [elem.split(', ')[0] for elem in elems]


def go_annotations(dburl, genus="Saccharomyces", species="cerevisiae"):
    """Accesses GO annotations in a MySQL database.

       Database data can be obtained from the geneontology website.

       example url: "mysql://username:password@hostname:port/go
    """
    go = Connect(dburl)
    g = go.species |Match(_.id,                   _.species_id)|      go.gene_product
    g = g          |Match(_.gene_product.id,      _.gene_product_id)| go.association
    g = g          |Match(_.association.term_id,  _.term2_id)|        go.graph_path
    g = g          |Match(_.term1_id,             _.id)|              go.term//"annot"
    g = g          |Match(_.relationship_type_id, _.id)|              go.term//"rel"
    g = g[(_.genus==genus) & (_.species == species)]
    return g.Get(_.symbol, _.annot.name/"annotation", _.rel.name/"relation_type", _.distance)%"annotations"
predefined_sources.register(go_annotations)


########################### #KEGG ######################################
@util.memoized
def kegg():
    from SOAPpy import WSDL
    wsdl = 'http://soap.genome.jp/KEGG.wsdl';
    serv = WSDL.Proxy(wsdl)
    return serv

def get_kegg_organisms():
    serv = kegg()
    return Rep(serv.list_organisms())
predefined_sources.register(get_kegg_organisms,name="organisms", category="kegg")

def get_kegg_pathways(org):
    serv = kegg()
    return Rep(serv.list_pathways(org))
predefined_sources.register(get_kegg_pathways,name="pathways", category="kegg")

def get_kegg_pathway(pathway):
    serv = kegg()
    relations = Rep(serv.get_element_relations_by_pathway(pathway))
    elements = Rep(serv.get_elements_by_pathway(pathway))
    return Combine(elements,relations)


    return Rep(serv.list_pathways(org))
predefined_sources.register(get_kegg_pathways,name="pathways", category="kegg")


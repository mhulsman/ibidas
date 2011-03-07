from ibidas.utils import util
from ibidas.utils.config import config
import os.path
import os
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

############################# YEAST ######################################
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
    res = res.To(_.start, _.stop, Do=_.Cast("int?"))
    res = res.To(_.genetic_pos,   Do=_.Cast("real64?"))
    return res.Copy()
predefined_sources.register(yeast_feats,name="genomic_feats",category="yeast")  


def yeast_aliases(feats, remove_multi=True):
    """Given a yeast feats table (see ``yeast_feats``), returns a 
       two column table with aliases and the corresponding  official feature names.
       All names in output are in uppercase.

       :param remove_multi: if True (default), removes aliases that map to multiple offical feature names.
       """

    res = Stack(feats.Get(_.gene_aliases, _.feat_name).Flat(), 
                feats.Get(_.gene_name, _.feat_name), 
                feats.Get(_.feat_name, _.feat_name))
    res = res/("alias", "feat_name")
    res = res%"feat_aliases"
    res = res[_.alias != ""].Unique()
    
    if remove_multi:
        multi_aliases = res.GroupBy(_.alias)[_.feat_name.Count() > 1].alias
        res = res[~(_.alias |In| multi_aliases)]
    
    res = res.Each(str.upper, dtype="bytes")
    return res.Copy()
predefined_sources.register(yeast_aliases,name="name_aliases",category="yeast")  


def yeast_phosphogrid():
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
predefined_sources.register(yeast_phosphogrid,name="phosphogrid",category="yeast")  


def yeast_kinome():
    url = "http://www.yeastkinome.org/files/supplementary/Website_Supplementary_Table_1_-_SAINT_Datasets.zip"
    res = Unpack(Fetch(url),subfiles='RAW/RAW-UNIQUE.txt')

    rtype = "[yeastkinome:*]<(bait_yorf=bytes, hit_yorf=bytes, bait_gene=bytes, hit_gene=bytes, saint=bytes)"
    res = Read(res,dtype=rtype,fieldnames=True)    
    res = res.To(_.saint, Do=_.Cast("real64"))
    return res
predefined_sources.register(yeast_kinome,name="yeast_kinome",category="yeast")  


def yeast_chipchip(translator=None):
    url = "http://fraenkel.mit.edu/improved_map/orfs_by_factor.tar.gz"
    res = Unpack(Fetch(url))
    dirname = os.path.dirname(res[0])
    pval_names = ['none', 'p0.005', 'p0.001']
    max_pvals = [1.0, 0.005, 0.001]
    cons = [0, 1, 2]

    loaded_data = []
    for mpval, pval_name in zip(max_pvals, pval_names):
        for con in cons:
            filename = os.path.join(dirname,"orfs_by_factor_%s_cons%s.txt" % (pval_name, str(con)))
            f = open(filename)
            results = []
            for row in f:
                res = row.split('\t')
                results.append((res[0], res[1:-1]))
            results = Rep(results).Flat()
            results = results.Get(_.f0/"trans_name",_.f1/"target", Rep(mpval)/"pval", Rep(con)/"conservation").Level(1)
            loaded_data.append(results)
    res = Stack(*loaded_data)%"chipchip"

    #group on tf,target tuple, keeping trans_name and target flat
    res = res.GroupBy((_.trans_name, _.target),flat=(_.trans_name, _.target))
    #take min pval found, and max conservation
    res = res.Get(_.trans_name, _.target, _.pval.Min(), _.conservation.Max())   
    if(not translator is None):
        res = res.To(_.trans_name, _.target, Do=_.Each(str.upper,dtype="bytes"))
        res = res.Replace(_.trans_name, translator)
        res = res.Replace(_.target, translator)
    return (res%"chipchip").Copy()
predefined_sources.register(yeast_chipchip,name="chipchip_macisaac",category="yeast")  
        
        
def yeastract(translator=None, url="http://www.yeastract.com/download/RegulationTwoColumnTable_Documented_20101213.tsv.gz"):
    """Downloads documented transcription factor regulation interactions from yeastract"""
    rtype = "[tftargets:*]<(trans_factor=bytes, target=bytes)"
    res = Read(Fetch(url),dtype=rtype)
    if(not translator is None):
        res = res.Each(str.upper,dtype="bytes")
        res = res.Replace(_.trans_factor, translator)
        res = res.Replace(_.target, translator)
    return (res%"yeastract").Copy()
predefined_sources.register(yeastract,category="yeast")

################################ HUMAN ##################################

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
    res = res.To(_.day, _.month, _.year, Do=_.Cast("int?"))
    res = res.To(_.mim, Do=_.Cast("int"))

    return res.Without(_.f8, _.f12, _.disease1, _.disease2, _.disease3).Copy()
predefined_sources.register(omim_genemap, category="human")

def omim_disease_parse(x):
    x = x.replace('{','').replace('}','')
    elems =  x.split('; ')
    return [elem.split(', ')[0] for elem in elems]


##################### GENERAL ################################
def in_memory_db():
    """Returns an empty in memory database"""
    return Connect("sqlite:///:memory:");
predefined_sources.register(in_memory_db)


@util.memoized
def string(dburl=config.get('databases.string_url',None)):
    return Connect(dburl)
predefined_sources.register(string,category="string")

def string_interactions(dburl=config.get('databases.string_url',None), species="Saccharomyces cerevisiae", subscores=False, external_names=False):
    """Given a Postgres db with String data, specified in dburl, and a species, returns all interactions and their score.

    The database data can be obtained from String.

    example url: "postgres://username:password@hostname:port/string_dbname"

    Use ``connect`` to access the whole database::
       
       #Get available species names: 
       >>> connect(dburl).items.species.offical_name

    """
    z = string(dburl)
    inter  = z.items.species |Match| z.network.protein_protein_links
    inter  = inter[_.official_name == species]
    inter  = inter |Match(_.protein_id_a, _.protein_id)| z.items.proteins//"left"
    inter  = inter |Match(_.protein_id_b, _.protein_id)| z.items.proteins//"right"
    if external_names:
        names = inter.Get(_.left.protein_external_id/"left", _.right.protein_external_id/"right").Each(lambda x : x.split('.')[1],dtype="bytes")
    else:
        names = inter.Get(_.left.preferred_name/"left", _.right.preferred_name/"right")

    if(subscores):
        return   inter.Get(names,
                        _.equiv_nscore/"neighborhood_score", _.equiv_nscore_transferred/"neighborhood_score_transferred", 
                        _.equiv_fscore/"fusion_score", 
                        _.equiv_pscore/"phylo_cooccurence_score", 
                        _.equiv_hscore/"homology_score", 
                        _.array_score/"coexpression_score", 
                        _.array_score_transferred/"coexpression_score_transferred",
                        _.experimental_score/"experimental_score",
                        _.experimental_score_transferred/"experimental_score_transferred", 
                        _.database_score/"curated_score", 
                        _.database_score_transferred/"curated_score_transferred", 
                        _.textmining_score/"textmining_score", 
                        _.textmining_score_transferred/"textmining_score_transferred",
                        _.combined_score) % "interactions"
    else:
        return   inter.Get(names, 
                        _.combined_score) % "interactions"
predefined_sources.register(string_interactions,name="interactions", category="string")

def string_interaction_types(dburl=config.get('databases.string_url',None), species="Saccharomyces cerevisiae", external_names=False):
    """Given a Postgres db with String data, specified in dburl, and a species, returns all interactions and their score.

    The database data can be obtained from String.

    example url: "postgres://username:password@hostname:port/string_dbname"

    Use ``connect`` to access the whole database::
       
       #Get available species names: 
       >>> connect(dburl).items.species.offical_name

    """
    z = string(dburl)
    inter  = z.network.actions
    inter  = inter |Match(_.item_id_a, _.protein_id)| z.items.proteins//"left"
    inter  = inter |Match(_.item_id_b, _.protein_id)| z.items.proteins//"right"
    inter = inter |Match(_.left.species_id, _.species_id)| z.items.species
    inter  = inter[_.official_name == species]
    if external_names:
        names = inter.Get(_.left.protein_external_id/"left", _.right.protein_external_id/"right").Each(lambda x : x.split('.')[1],dtype="bytes")
    else:
        names = inter.Get(_.left.preferred_name/"left", _.right.preferred_name/"right")
    return   inter.Get(names,
                    _.mode, _.action, _.a_is_acting,
                    _.score) % "interactions"
predefined_sources.register(string_interaction_types,name="interaction_types",category="string")


@util.memoized
def open_go(dburl=config.get("databases.go_url",None)):
    go = Connect(dburl)
    return go
predefined_sources.register(open_go,name="go",category="go")


def go_annotations(dburl=config.get("databases.go_url",None), genus="Saccharomyces", species="cerevisiae"):
    """Accesses GO annotations in a MySQL database.

       Database data can be obtained from the geneontology website.

       example url: "mysql://username:password@hostname:port/go
    """
    go = open_go(dburl)
    g = go.species |Match(_.id,                   _.species_id)|      go.gene_product
    g = g          |Match(_.gene_product.id,      _.gene_product_id)| go.association
    g = g          |Match(_.association.term_id,  _.term2_id)|        go.graph_path
    g = g          |Match(_.term1_id,             _.id)|              go.term//"annot"
    g = g          |Match(_.relationship_type_id, _.id)|              go.term//"rel"
    g = g          |Match(_.association.id,       _.association_id)|  go.evidence
    g = g[(_.genus==genus) & (_.species == species)][_.is_not == False]
    return g.Get(_.symbol, _.annot.name/"annotation", _.rel.name/"relation_type", _.evidence.code, _.distance)%"annotations"
predefined_sources.register(go_annotations,name="annotations",category="go")


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
    relations = Rep(serv.get_element_relations_by_pathway(pathway))%"relations"
    elements = Rep(serv.get_elements_by_pathway(pathway))%"nodes"
    pwdata = Combine(elements,relations)
    
    names = pwdata.names.Flat().Unique()()
    name_info = []
    while len(names) > 0:
        xnames = names[:100]
        names = names[100:]
        res = serv.btit(" ".join(xnames))
        for elem in res.split('\n'):
            if(elem):
                pos = elem.index(' ')
                name_info.append((elem[:pos], elem[(pos+1):]))
    res = (pwdata |Match(jointype="left")| Rep(name_info)/("names","info")).Without(_.R.names).Copy()
    #info = res.Get((_.names, _.info)).Each(extract_info,dtype="bytes")/"info"
    #res = res.Without(_.info).Get("*", info)
    return res
predefined_sources.register(get_kegg_pathway,name="pathway", category="kegg")


def extract_info(vals):
    names, info = vals
    res = str(info).split('; ')[0]
    if res == "--":
        res = ""
    return res


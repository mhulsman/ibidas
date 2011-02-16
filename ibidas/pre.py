_delay_import_(globals(),"ibidas","*")

class Pre(object):
    def register(self, func):
        setattr(self,func.__name__, func)
predefined_sources = Pre()


def yeast_feats():
    """Returns table of yeast genome features,
       from:  http://downloads.yeastgenome.org/chromosomal_feature/SGD_features.tab
    """

    rtype = "[feats:*]<(sgdid=bytes, feat_type=bytes, feat_qual=bytes, feat_name=bytes, gene_name=bytes, gene_aliases=bytes, feat_parent_name=bytes, sgdid_alias=bytes, chromosome=bytes, start=bytes, stop=bytes, strand=bytes[1], genetic_pos=bytes, coordinate_version=bytes[10], sequence_version=bytes, description=bytes)"
    
    res = read(download.get("http://downloads.yeastgenome.org/chromosomal_feature/SGD_features.tab"),dtype=rtype)

    splitfunc = lambda x: x.split("|")
    outtype = "[aliases:~]<bytes"
    res = res.to(_.gene_aliases,  do=_.each(splitfunc, dtype=outtype).elements()[_ != ""])
    res = res.to(_.start, _.stop, do=_.cast("int$"))
    res = res.to(_.genetic_pos,   do=_.cast("real64$"))
    return res.copy()
predefined_sources.register(yeast_feats)  


def yeast_aliases(feats):
    """Given a yeast feats table (see ``yeast_feats``), returns a 
       two column table with the official feature names and a column with aliases.

       Can be used to map names to official names (with match)"""

    res = stack(feats.get(_.feat_name, _.gene_aliases).flat(), 
                feats.get(_.feat_name, _.gene_name), 
                feats.get(_.feat_name, _.feat_name))
    res = res[_.alias != ""]
    res = res%"feat_aliases"
    res = res/("feat_name", "alias")
    return res.tuple().unique().attributes().copy()
predefined_sources.register(yeast_aliases)  

def in_memory_db():
    """Returns an empty in memory database"""
    return connect("sqlite:///:memory:");
predefined_sources.register(in_memory_db)


def yeastract(url="http://www.yeastract.com/download/RegulationTwoColumnTable_Documented_20101213.tsv.gz"):
    """Downloads documented transcription factor regulation interactions from yeastract"""
    rtype = "[tfs:*]<(tf=bytes, target=bytes)"
    res = read(download.get(url),dtype=rtype)
    return res.copy()
predefined_sources.register(yeastract)


def string_interactions(dburl, species="Saccharomyces cerevisiae"):
    """Given a Postgres db with String data, specified in dburl, and a species, returns all interactions and their score.

    The database data can be obtained from String.

    example url: "postgres://username:password@hostname:port/string_dbname"

    Use ``connect`` to access the whole database::
       
       #Get available species names: 
       >>> connect(dburl).items.species.offical_name

    """
    z = connect(dburl)
    inter = z.items.species.match(z.network.protein_protein_links)
    inter = inter[_.official_name == species]
    pyeast = z.items.proteins 
    inter = inter.match(pyeast//"left",_.protein_id_a, _.protein_id)
    inter = inter.match(pyeast//"right",_.protein_id_b, _.protein_id)
    return inter.get(_.Bleft.preferred_name/"left", _.Bright.preferred_name/"right", _.combined_score)%"interactions"
predefined_sources.register(string_interactions)


def omim_genemap():
    """The omim genemap data"""
    rtype = "[omim:*]<(chr_map_entry_nr=bytes, month=bytes, day=bytes, year=bytes, location=bytes, gene_symbol=bytes, gene_status=bytes[1], title=bytes, f8=bytes, mim=bytes, method=bytes, comments=bytes, f12=bytes, disease1=bytes, disease2=bytes, disease3=bytes, mouse=bytes, reference=bytes)"
    res = read(download.get("ftp://ftp.ncbi.nih.gov/repository/OMIM/genemap"),dtype=rtype)
    res = res.to(_.day, _.month, _.year, do=_.cast("int$"))
    
    splitfunc = lambda x: x.split(", ")
    outtype = "[symbols:~]<bytes"
    res = res.to(_.gene_symbol,  do=_.each(splitfunc, dtype=outtype).elements()[_ != ""])
    
    outtype = "[methods:~]<bytes"
    res = res.to(_.method,  do=_.each(splitfunc, dtype=outtype).elements()[_ != ""])
    
    res = res.get( _.get(_.disease1, _.disease2, _.disease3).harray("diseases").elements()[_ != " "]/"disease", "~")
    return res.without(_.f8, _.f12, _.disease1, _.disease2, _.disease3).copy()
predefined_sources.register(omim_genemap)


def go_annotations(dburl, genus="Saccharomyces", species="cerevisiae"):
    """Accesses GO annotations in a MySQL database.

       Database data can be obtained from the geneontology website.

       example url: "mysql://username:password@hostname:port/go
    """
    go = connect(dburl)
    g = go.species
    g = g.match(go.gene_product, _.id,                   _.species_id)
    g = g.match(go.association,  _.Bgene_product.id,     _.gene_product_id)
    g = g.match(go.graph_path,   _.Bassociation.term_id, _.term2_id)
    g = g.match(go.term,         _.term1_id,             _.id)
    g = g[(_.genus==genus) & (_.species == species)]
    return g.get(_.symbol, _.Bterm.name, _.relationship_type_id, _.distance)
predefined_sources.register(go_annotations)

    

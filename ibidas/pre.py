_delay_import_(globals(),"ibidas","*")

class Pre(object):
    def register(self, func):
        setattr(self,func.__name__, func)
predefined_sources = Pre()


def yeast_feats():
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
    res = stack(feats.get(_.feat_name, _.gene_aliases).flat(), 
                feats.get(_.feat_name, _.gene_name), 
                feats.get(_.feat_name, _.feat_name))
    res = res[_.alias != ""]
    res = res%"feat_aliases"
    res = res/("feat_name", "alias")
    return res.tuple().unique().attributes().copy()
predefined_sources.register(yeast_aliases)  

def in_memory_db():
    return connect("sqlite:///:memory:");
predefined_sources.register(in_memory_db)


def yeastract(url="http://www.yeastract.com/download/RegulationTwoColumnTable_Documented_20101213.tsv.gz"):
    rtype = "[tfs:*]<(tf=bytes, target=bytes)"
    res = read(download.get(url),dtype=rtype)
    return res.copy()
predefined_sources.register(yeastract)

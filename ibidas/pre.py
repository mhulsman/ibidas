_delay_import_(globals(),"ibidas","*")

class Pre(object):
    def register(self, func):
        setattr(self,func.__name__, func)
predefined_sources = Pre()


def yeast_orfs():
    rtype = "[orfs:*]<(sgdid=bytes, feat_type=bytes, feat_qual=bytes, feat_name=bytes, gene_name=bytes, gene_aliases=bytes, feat_parent_name=bytes, sgdid_alias=bytes, chromosome=bytes, start=int32$, stop=int32$, strand=bytes[1], genetic_pos=real64$, coordinate_version=bytes[10], sequence_version=bytes, description=bytes)"
    res = read(download.get("http://downloads.yeastgenome.org/chromosomal_feature/SGD_features.tab"),dtype=rtype)

    splitfunc = lambda x: x.split("|")
    outtype = "[aliases:~]<bytes"
    res = res.to(_.gene_aliases, do=_.each(splitfunc, dtype=outtype).elements())
    return res.copy()
predefined_sources.register(yeast_orfs)  

def in_memory_db():
    return connect("sqlalchemy:///:memory:");
predefined_sources.register(in_memory_db)




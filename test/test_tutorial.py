import unittest
import numpy

from ibidas.utils import util
from ibidas import *


class TestTutorial(unittest.TestCase):
    
    def test_yeastract_inoutdegree(self):
        url = "http://www.yeastract.com/download/RegulationTwoColumnTable_Documented_20101213.tsv.gz"
        yeastract = Read(Fetch(url))
        
        str(yeastract)
        yeastract = yeastract/('trans_factor','target')
        yeastract = yeastract%'tftargets'
        str(yeastract)
        yeastract = yeastract.Rename('trans_factor','target')
        yeastract = yeastract.DimRename('tftargets')
        str(yeastract)
        yeastract = yeastract.Copy()
        
        gyeastract = yeastract.GroupBy(_.trans_factor)
        gyeastract = yeastract.GroupBy(yeastract.trans_factor)
        str(gyeastract)
        
        gyeastract = gyeastract.Get(_.trans_factor, _.target.Count()/"out_degree")
        str(gyeastract)

        yeastract = Get.yeast.yeastract()
        yeastract.GroupBy(_.trans_factor).target.Count()()
        yeastract.GroupBy(_.target).trans_factor.Count()()



    def test_yeastract_inoutdegreeb(self):
        def yeastract2(url="http://www.yeastract.com/download/RegulationTwoColumnTable_Documented_20101213.tsv.gz"):
            """Downloads documented transcription factor regulation interactions from yeastract"""

            res = Read(Fetch(url),dtype="[tftargets:*]<(trans_factor=bytes, target=bytes)")
            return res.Copy()

        Get.register(yeastract2)



    def test_tutorial_chromodist(self):
        rtype = """[feats:*]<(sgdid=bytes, feat_type=bytes, feat_qual=bytes, feat_name=bytes, gene_name=bytes,
                gene_aliases=bytes, feat_parent_name=bytes, sgdid_alias=bytes, chromosome=bytes,
                start=bytes, stop=bytes, strand=bytes[1], genetic_pos=bytes, coordinate_version=bytes[10],
                sequence_version=bytes, description=bytes)"""

        res = Read(Fetch("http://downloads.yeastgenome.org/chromosomal_feature/SGD_features.tab"),dtype=rtype)

        resx = res/{'f3': 'feat_name', 'f8':'chromosome', 'f9':'start'}
        str(resx)

        res = res.To(_.start, _.stop, Do=_.Cast("int?"))
        res = res.To(_.genetic_pos,   Do=_.Cast("real64?"))
        
        resy = res.To(_.gene_aliases,  Do=_.Each(_.split('|')).Elem()[_ != ""])


        splitfunc = lambda x: x.split('|')
        resx = res.To(_.gene_aliases,  Do=_.Each(splitfunc, dtype="[aliases:~]<bytes").Elem()[_ != ""])
        
        yeast_feats = resy.Copy()
        yeast_feats = Get.yeast.genomic_feats()

        yeastract = Get.yeast.yeastract()
        yeast_feats = yeast_feats.To(_.feat_name, Do=_.Each(str.upper))
        yeastract = yeastract.To(_.target,         Do=_.Each(str.upper))

        tf_feat = yeastract.Match(yeast_feats, _.target, _.feat_name)
        str(tf_feat)

        self.assertTrue((yeastract.target.Count() - tf_feat.target.Count()) == 72)

        nonmatched = (yeastract.target.Set() - tf_feat.target.Set()).Elem()
        str(nonmatched)


        str(nonmatched.In(yeast_feats.gene_name.Each(str.upper)))
        str(nonmatched |In| yeast_feats.gene_name.Each(str.upper))
        str(nonmatched |In| yeast_feats.gene_aliases.Each(str.upper))
        str(Any(nonmatched |In| yeast_feats.gene_aliases.Each(str.upper)))
        str(nonmatched |In| yeast_feats.gene_aliases.Flat().Each(str.upper))
       
        nonmatched_feats = nonmatched.Match(yeast_feats.Flat(), _.target, _.gene_aliases.Each(str.upper))
        str(nonmatched_feats)


        str(tf_feat.Unique(_.trans_factor, _.target).target.Count())

        nonmatched_unique = nonmatched_feats.GroupBy(_.target)[Count(_.feat_name) == 1].target
        str(nonmatched_unique)

        yt_nm = yeastract[_.target |In| nonmatched_unique]

        yeast_feats = yeast_feats.Get("*", _.gene_aliases/"gene_aliases2")
        yeast_feats = yeast_feats.To(_.gene_aliases, Do=_.Array())
        res = yt_nm.Match(yeast_feats.Flat(), _.target, _.gene_aliases2).Without(_.gene_aliases2)
        
        str(res)
        tf_feat = Stack(tf_feat, res).Copy()
        Save(tf_feat, 'tf_feat.dat')
        tf_feat = Load('tf_feat.dat')


        tf_feat = tf_feat.GroupBy(_.trans_factor, _.chromosome)
        res = tf_feat.Get(_.trans_factor, _.chromosome, _.sgdid.Count()/"count", _.start).Copy()
        str(res)

        str(Corr(res.count))
        str(Corr(res.count.Transpose()))

        res = res.To(_.chromosome, Do=_.Cast("int?"))

        str(res.Sort(_.chromosome).Get(_.chromosome, Corr(_.count.Transpose()/"chromo_corr")))
        str(res.Get(_.chromosome, _.count.Sum("gtrans_factor")))
        str(res.count.Sort().Sum("gtrans_factor"))
        normalized_counts = res.count.Cast("real64") / res.count.Sum("gtrans_factor")
        str(normalized_counts.Sort().Sum("gtrans_factor"))


        normalized_counts.Sort().Sum("gtrans_factor")()







        









        









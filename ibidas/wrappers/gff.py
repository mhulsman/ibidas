from .. import repops_multi;
from python import Rep;
from ..utils import util

entry_field_names = [ 'seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attr'];
entry_field_types = [ str,       str,      str,       int,     int,   str,     str,      str,     str   ];

def read_gff3(fname, **kwargs):

    gff3_fieldnames = ( 'seqname', 'source', 'id', 'parent', 'name', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attr' );

    f       = util.open_file(fname,mode='rU');
    entries = [];

    i = 0;
    for line in f:
        i = i + 1;

        line = line.strip();
        if line[0] in [ '#', '%' ]:
            continue;
        #fi
        entry = split_entry(line);

        entries.append(entry);
    #efor

    loci = [];

    for entry in entries:
      e = [ entry[field] if field in entry else '' for field in gff3_fieldnames ];
      e[-1] = ';'.join([ '='.join(attr) for attr in e[-1].items() ]);
      loci.append(tuple(e));
    #efor

    R = Rep(loci) / gff3_fieldnames;
    
    for (name, type) in zip(entry_field_names, entry_field_types):
      R = R.To(name, Do=_.Cast(type));
    #efor

    return R.Copy();
#edef




    

def split_entry(raw):

    fields = raw.split('\t', len(entry_field_names));

    entry           = dict( zip(entry_field_names, fields ) );
    entry['attr']   = dict(tuple(attr.split('=')) for attr in entry['attr'].split(';'));
    entry['id']     = entry['attr']['ID'] if 'ID' in entry['attr'] else '';
    entry['parent'] = entry['attr']['Parent'] if 'Parent' in entry['attr'] else '';
    entry['name']   = entry['attr']['Name'] if 'Name' in entry['attr'] else '';

    if 'ID' in entry['attr']:
      del entry['attr']['ID'];
    #fi

    if 'Parent' in entry['attr']:
      del entry['attr']['Parent'];
    #fi

    if 'Name' in entry['attr']:
      del entry['attr']['Name'];
    #fi

    return entry;

#edef
    


def save_gff3(data, filename, seqid, source='unknown'):
    """
    Saves data in gff3 format. 
    :param seqid: name of e.g. chromosome described by dataset
    :param source: source of data
    :param data: Required format = locus id, [type], [[start]], [[stop]], [[strand]], [attr], <[score]>, <[phase]>. Here, the [] signs indicate nested dimensions. Score and phase are optional.

    For example, to get to this format from the first record in a genbank/embl parsed file, use:
    data = Read('genbankfile.gb')
    data[0].Get(_.attr.locus_tag, _.feat_key, _.start, _.stop, _.strand, _.attr).GroupBy(_.locus_tag)

    """
    file = open(filename,'w')

    #sort on start position for all records
    data = data.Get('*', _.Get(2).Min().Min()/'min_start', _.Get(3).Max().Max()/'max_stop').Sort(_.min_start)    

    #sort on start position within loci
    data = data.Sort(_.Get(3).Min())    

    #create dictionary records
    data = data.Dict().Each(to_row).Elems().Fields().Copy()

    #add seqid, source columns
    data = repops_multi.Combine(python.Rep((seqid, source)), data).FlatAll()
   
    data = data.Tuple()()
    for row in data:
        line = '\t'.join([str(elem) for elem in row])
        file.write(line + '\n')
    file.close()


def to_row(record):
    result = []
    locus_tag = record['locus_tag']
    feat_keys = record['feat_key']
    start = record['start']
    stop = record['stop']
    strand = record['strand']
    rattr = record['attr']
    if 'score' in record:
        score = record['score']
    else:
        score = ['.'] * len(rattr)
    if 'phase' in record:
        phase = record['phase']
    else:
        phase = [0] * len(rattr)
   
    gene = None
    exons = []
    cdss = []
    mrnas = []
    others = []
    for pos, fkey in enumerate(feat_keys):
        if fkey == 'gene':
            assert gene is None, 'Multiple genes per locus not supported!'
            assert len(start[pos]) == 1, 'Only one region per gene supported!'
            gene = pos
        elif fkey == 'exon':
            exons.append(pos)
        elif fkey == 'CDS':
            cdss.append(pos)
        elif fkey == 'mRNA':
            mrnas.append(pos)
        else:
            others.append(pos)


    gid = locus_tag
    if gene is None: #no gene found, generate from min/max pos
        if mrnas or cdss or exons:
            gene_strand = '+' if strand[(mrnas + cdss + exons + others)[0]][0] else '-'
            result.append(('gene', record['min_start'], record['max_stop'], '.', gene_strand, '.', 'ID=' + gid))
    else:
        gene_strand = '+' if strand[gene][0] else '-'
        if mrnas or cdss or exons:
            gene_strand_other = '+' if strand[(mrnas + cdss + exons + others)[0]][0] else '-'
            if gene_strand != gene_strand_other:
                util.warning('Strand of gene %s does not match strand of mRNA/CDS/exons' % gid)
                gene_strand = gene_strand_other
        attr = {'ID':gid}
        attr = process_attr('gene', attr, rattr[gene])
        result.append(('gene', start[gene][0], stop[gene][0], score[gene], gene_strand, '.', attr))
        
    #generate exon cache
    exon_dict = dict()
    has_exon_info = dict()
    exon_parent = dict()
    exon_strand_dict = dict()
    for pos, exon_pos in enumerate(exons):
        eid = locus_tag + '_exon' + str(pos)
        n = zip(start[exon_pos], stop[exon_pos])
        assert len(n) == 1, 'Cannot have exon with disjoint region'
        exon_dict[n[0]] = eid
        exon_parent[eid] = []
        has_exon_info[eid] = exon_pos

    #generate exons from mrnas
    for pos, mrna_pos in enumerate(mrnas):
        mid = locus_tag + '_mRNA' + str(pos)
        locpairs = zip(start[mrna_pos], stop[mrna_pos])
        min_start = start[mrna_pos].min()
        max_stop = stop[mrna_pos].max()

        attr = {'ID':mid, 'Parent':gid}
        attr = process_attr('mRNA',attr, rattr[mrna_pos])
        mRNA_strand = '+' if strand[mrna_pos][0] else '-'
        result.append(('mRNA', min_start, max_stop, score[mrna_pos], mRNA_strand, '.', attr))

        for epos, locpair in enumerate(locpairs):
            if not locpair in exon_dict:
                eid = locus_tag + '_exon' + str(len(exon_dict))
                exon_dict[locpair] = eid
                exon_parent[eid] = []
            eid = exon_dict[locpair]
            exon_parent[eid].append(mid)
            exon_strand_dict[eid] = strand[mrna_pos][epos]

    #write exon records
    exon_list = list(exon_dict.iteritems())
    exon_list.sort(key = lambda x: x[1])
    for locpair, eid in exon_list:
       if eid in has_exon_info:
           exon_pos = has_exon_info[eid]
           exon_strand = '+' if strand[exon_pos][0] else '-'
           attr = {'ID':eid, 'Parent':','.join(exon_parent[eid])}
           attr = process_attr('exon', attr, rattr[exon_pos])
           result.append(('exon', locpair[0], locpair[1], score[exon_pos], exon_strand, '.', attr))
       else:
           attr = process_attr('exon', {'ID':eid, 'Parent':','.join(exon_parent[eid])},{})
           exon_strand = '+' if exon_strand_dict[eid] else '-'
           result.append(('exon', locpair[0], locpair[1], '.', exon_strand, '.', attr))  
    
    #generate CDSss from mrnas
    for cds_pos in cdss:
        cid = locus_tag + '_CDS' + str(pos)
        locpairs = zip(start[cds_pos], stop[cds_pos])
        
        attr = {'ID':cid, 'Parent':mid}#FIXME
        attr = process_attr('CDS', attr, rattr[mrna_pos])
        CDS_strand = '+' if strand[cds_pos][0] else '-'

        for epos, locpair in enumerate(locpairs):
            result.append(('CDS', locpair[0], locpair[1], score[cds_pos], CDS_strand, phase[cds_pos], attr))
      
    for pos, other_pos in enumerate(others):
        oid = locus_tag + feat_keys[other_pos] + str(pos)
        n = zip(start[other_pos], stop[other_pos])
        assert len(n) == 1, 'Cannot have other regions with multiple regions'
        
        attr = {'ID':oid}
        if not gene is None:
            attr['Parent'] = gid

        attr = process_attr('other',attr, rattr[other_pos])
        result.append((feat_keys[other_pos], start[other_pos][0], stop[other_pos][0], record.get('score','.'), mRNA_strand, '.', attr))
        

    return result

def process_attr(key, attr, extraattr):
    attr.update({})
    res = []
    for key,value in attr.iteritems():
        res.append(str(key) + '=' + str(value))
    return ';'.join(res)

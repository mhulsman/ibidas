from ..utils import util;
from fasta import write_fasta_text
import numpy
import tempfile
import csv
import math
import os
import fnmatch

###############################################################################

r_qlen   = lambda x: x[0]
r_qstart = lambda x: x[1]
r_qend   = lambda x: x[2] 
r_slen   = lambda x: x[3] 
r_sstart = lambda x: x[4] 
r_send   = lambda x: x[5]
r_length = lambda x: x[6] 
r_mismatch = lambda x: x[7] 
r_gapopen = lambda x: x[8]
r_pident = lambda x: x[9] 
r_evalue = lambda x: x[10] 
r_bitscore = lambda x: x[11];

def blast(data, type, folder, reciprocal = True, normalize = False, overwrite = False, blastopts=''):

  seq_1 = data[0];
  seq_2 = data[1];

  title_1 = [ "%d" % i for i in xrange(len(seq_1)) ];
  title_2 = [ "%d" % i for i in xrange(len(seq_2)) ];

  fas_1 = tempfile.NamedTemporaryFile(delete = False);
  fas_2 = tempfile.NamedTemporaryFile(delete = False);
  db_1  = "%s.blastdb" % (fas_1.name);
  db_2  = "%s.blastdb" % (fas_2.name)

  md5_1 = write_fasta_text(title_1, seq_1, len(seq_1), fas_1);
  md5_2 = write_fasta_text(title_2, seq_2, len(seq_2), fas_2);
  fas_1.close();
  fas_2.close();

  mkdb_CMDs = [];
  blst_CMDs = [];

    # perform blast for 12
  file_12 = "%s/%s-%s.tsv" % (folder, md5_1, md5_2);
  mkdb_CMDs = mkdb_CMDs + [ blast_make_db_CMD(fas_2.name, db_2, type) ];
  blst_CMDs = blst_CMDs + [ blast_run_CMD(fas_1.name, db_2, type, file_12, blastopts, overwrite) ];

  if reciprocal:
      # perform blast for 21
    file_21 = "%s/%s-%s.tsv" % (folder, md5_2, md5_1);
    mkdb_CMDs = mkdb_CMDs + [ blast_make_db_CMD(fas_1.name, db_1, type) ];
    blst_CMDs = blst_CMDs + [ blast_run_CMD(fas_2.name, db_1, type, file_21, blastopts, overwrite) ];
  #fi

  if normalize:
      # perform blast for 11
    file_11 = "%s/%s-%s.tsv" % (folder, md5_1, md5_1);
    blst_CMDs = blst_CMDs + [ blast_run_CMD(fas_1.name, db_1, type, file_11, blastopts, overwrite) ];

      # perform blast for 22
    file_22 = "%s/%s-%s.tsv" % (folder, md5_2, md5_2);
    blst_CMDs = blst_CMDs + [ blast_run_CMD(fas_2.name, db_2, type, file_22, blastopts, overwrite) ];
  #fi

  util.run_par_cmds(mkdb_CMDs);
  util.run_par_cmds(blst_CMDs);

  del_CMDs = [ "rm -f '/tmp/%s'" % f for f in os.listdir('/tmp') if (fnmatch.fnmatch(f, "*%s*" % fas_1.name.split('/')[2])) or  (fnmatch.fnmatch(f, "*%s*" % fas_2.name.split('/')[2])) ];
  util.run_seq_cmds(del_CMDs);

  ab = blast_res_to_dict(file_12); # if reciprocal blast_reciprocal(file_12.name, file_21.name) else blast_res_to_dict(file_12.name)

  if reciprocal:
    ba = blast_res_to_dict(file_21);
    ab = blast_reciprocal(ab, ba);
  #fi

  if normalize:
    aa = blast_res_to_dict(file_11, max=True);
    bb = blast_res_to_dict(file_22, max=True);
    ab = blast_bitscore_normalize(ab, aa, bb);
  #fi

    #         qseqid  sseqid qlen qstart qend slen sstart send length mismatch gapopen pident evalue bitscore
  sp_types = (int,    int,    int,  int, int, int,  int,    int,  int,   int,      int,     float,  float,  float)
  ab = [ [ list(p[0]) + h for h in p[1] ] for p in ab.items() ];
  ab = [ item for sublist in ab for item in sublist];

  return tuple([ util.darray(row,type) for (type,row) in zip( sp_types, map(lambda *row: list(row), *ab)) ] );

#edef

###############################################################################

def blast_res_to_dict(blast_res, max=False):

  bm = {};
  br = open(blast_res, 'r');

    #         qseqid  sseqid qlen qstart qend slen sstart send length mismatch gapopen pident evalue bitscore
  sp_types = (int,    int,    int,  int, int, int,  int,    int,  int,   int,      int,     float,  float,  float)

  rdr = csv.reader(br, delimiter='\t', quotechar='"');
  for row in rdr:
    row = [ sp_types[i](row[i]) for i in xrange(len(row)) ];
    k = (row[0], row[1]);
    if max == True:
      if k in bm:
        bm[k] = [row[2:]] if row[-1] > bm[k][0][-1] else bm[k];
      else:
        bm[k] = [row[2:]];
      #fi
    else:
      if k in bm:
        bm[k] = bm[k] + [row[2:]];
      else:
        bm[k] = [row[2:]];
      #fi
    #fi
  #efor
  br.close();
  return bm;
#edef

###############################################################################

def blast_reciprocal(ab, ba):

  rab = {}

  for kab in ab.keys():
    if  not(kab[::-1] in ba):
      continue;
    #fi
    m = blast_reciprocal_match(ab[kab], ba[kab[::-1]]);
    if m:
      rab[kab] = m;
    #fi

  return rab;
#edef

###############################################################################

def blast_reciprocal_match(ab, ba):

  m = [];

  ab = sorted(ab, key=r_qstart)
  ba = sorted(ba, key=r_sstart)

  i = 0;
  j = 0;

  while i < len(ab) and j < len(ba):
    if r_qstart(ab[i]) > r_send(ba[j]):
      j = j + 1
    elif r_sstart(ba[j]) > r_qend(ab[i]):
      i = i + 1
    elif (r_qstart(ab[i]) == r_sstart(ba[j]) and r_qend(ab[i]) == r_send(ba[j])) or \
         (blast_hit_overlap(r_qstart(ab[i]), r_qend(ab[i]), 
                            r_sstart(ba[j]), r_send(ba[j]), 
                            r_length(ab[i]), r_length(ba[j])) > 0.6):
      m = m + [ab[i]]
      i = i + 1
      j = j + 1
    else:
      j = j + 1
    #fi
  #efor

  return m;
#edef

###############################################################################

def blast_hit_overlap(s1,e1, s2, e2, l1, l2):
  ov = 0
  if (s2 > e1) or (s1 > e2):
    ov = 0;
  elif (s2 >= s1) and (e2 >= e1):
    ov = e1 - s2
  elif (s1 >= s2) and (e1 >= e2):
    ov = e2 - s1
  elif (s1 >= s2) and (e2 >= e1):
    ov = e1 - s1
  elif (s2 >= s1) and (e1 >= e2):
    ov = e2 - s2
  #fi

  return float(ov) / float(min(l1, l2));
#edef

###############################################################################

def blast_bitscore_normalize(ab, aa, bb):
  # Normalize the bitscore

  nbm = {};

  for (kq, kd) in ab.keys():
    nabk = [];
    for h in ab[(kq,kd)]:
      nabk = nabk + [ h[0:-1] + [ float(h[-1]) / math.sqrt(float(aa[(kq,kq)][0][-1]) * float(bb[(kd,kd)][0][-1])) ] ];
    #efor
    nbm[(kq,kd)] = nabk;
  #efor

  return nbm;
#edef

###############################################################################

def blast_make_db_CMD(fas_file, filename, type):
  #makeblastdb -in "fas_file" -out "filename" -dbtype "prot"
  return "makeblastdb -in '%s' -out '%s' -dbtype '%s'" % (fas_file, filename, type);
#edef

###############################################################################

def blast_run_CMD(query, db, type, filename, blastopts, overwrite):
  prog = "blastp" if (type == 'prot') else "blastn";
  #       0      1      2    3      4    5    6      7    8      9        10      11     12     13
  opts = "qseqid sseqid qlen qstart qend slen sstart send length mismatch gapopen pident evalue bitscore"
  if (not os.path.isfile(filename)) or (overwrite == True):
    return "%s -query '%s' -db '%s' -out '%s' -outfmt '6 %s' %s" % (prog, query, db, filename, opts, blastopts);
  #fi

  return ""
#edef

###############################################################################

#from ibidas_blast import *;
#x = Read(Fetch("ftp://ftp.ensembl.org/pub/release-70/fasta/homo_sapiens/pep/Homo_sapiens.GRCh37.70.pep.abinitio.fa.gz"), sep=[' '])
#seq1 = x.f5[00:10]();
#seq2 = x.f5[10:20]();
#r = blast(seq1, seq2, 'prot', '/home/nfs/thiesgehrmann/.ibidas/blasts', overwrite=True, normalize=True)

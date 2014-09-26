from python import Rep;
from ..utils import util;

##############################################################################

sam_fieldnames = ('qname', 'flag', 'rname', 'pos', 'mapq', 'cigar', 'rnext', 'pnext', 'tlen', 'seq', 'qual', 'optional');
sam_fieldtypes = (str,     int,    str,     int,   int,    str,     str,     int,     int,    str,   str,    str);
max_fieldnames = len(sam_fieldnames) - 1;

##############################################################################

def read_sam(fname, **kwargs):
    f   = util.open_file(fname,mode='rU');
    als = [];
    hds = "";
    i   = 0;

    maxfields = max_fieldnames;

    for line in f:
      i    = i+1;
      line = line.strip();
      if line[0] == '@':
        hds = hds + line + '\n';
        continue;
      #fi

      fields    = line.split('\t', max_fieldnames);
      fieldslen = len(fields);
      maxfields = maxfields if maxfields > fieldslen else fieldslen;

      print line;
      print fieldslen, len(sam_fieldnames);

      if fieldslen < max_fieldnames:
        util.warning('Line %d does not have all the mandatory fields. Skipping' % i);
      else:
        als.append(tuple(fields));
      #fi
    #efor

    f.close();

    R = Rep(als) / sam_fieldnames[0:maxfields];

    for (name, type) in zip(sam_fieldnames, sam_fieldtypes):
      R = R.To(name, Do=_.Cast(type));
    #efor

    R.__dict__['sam_hds'] = hds;

    return R;
#edef

##############################################################################

def write_sam(data, filename, **kwargs):
  save_sam_rep(data, open(filename, 'w'));
#edef

##############################################################################

def save_sam_rep(data, fout):


  import md5;

  if len(data.Names) < len(sam_fieldnames) - 1:
    util.error("This structure doesn't have the correct number of fields. It has %d when it should have at least %d" % ( len(data.Names), len(sam_fieldnames) - 1));
  #fi
  if data.Names[0:len(sam_fieldnames)-1] != sam_fieldnames[0:-1]:
    util.warning("The field names in this structure are not as expected for a SAM structure. Are you sure it is the right format?");
  #fi

  m = md5.new();

  hds = {};
  if 'sam_hds' in data.__dict__:
    hds = data.sam_hds;
  #fi

  fout.write(hds);
  m.update(hds);

  lines = zip(*data());

  for line in lines:
    line = '\t'.join(line) + '\n';
    fout.write(line);
    m.update(line);
  #efor

  return m.hexdigest();
    
############################################################################## 
  
  

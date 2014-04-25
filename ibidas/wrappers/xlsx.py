from python import Rep;
from ..utils import util;

import xlsxwriter as xw;

################################################################################

def write_xlsx(R, filename, **kwargs):

  formats = kwargs.pop('formats', { None : {} });

  wb = xw.Workbook(filename);
  ws = wb.add_worksheet();

  formats[None] = {};
  formats = dict([(name,wb.add_format(fmt)) for (name,fmt) in formats.items()]);

  if "format" in R.Names:
    row_formats = R.format();
    R = R.Without(_.format);
  else:
    row_formats = [None] * R.Shape()();
  #fi

  R = R.ReplaceMissing();
  D = zip(*R());

  row = 0;
  for (l, fmt) in zip(D, row_formats):
    col = 0;
    if fmt not in formats:
      fmt = None;
    #fi
    
    for el in l:
      ws.write(row, col, el, formats[fmt]);
      col = col + 1;
    #efor
    
    row = row + 1;
  #efor

  wb.close();

#edef

################################################################################


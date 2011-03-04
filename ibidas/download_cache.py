import urllib2
import urlparse
import os
from os import path
from distutils.dir_util import mkpath

import tarfile
import zipfile
import gzip
import bz2

class DownloadCache(object):
    def __init__(self, folder=None):
        if(folder is None):
            folder = path.expanduser("~") + "/.ibidas/data"
        self.set_download_folder(folder)

    def url_to_file(self, url):
        return url.split('/')[-1]

    def __call__(self, url, dbname=None, filename=None):
        if(filename is None):
            filename = self.url_to_file(url)
        if(dbname is None):
            dbname = urlparse.urlparse(url).netloc

        file_path = self.get_filename(dbname, filename) 
        if not path.exists(file_path):
            self.download(url, file_path)
        
        return file_path

    def set_download_folder(self, folder):
        self.folder = folder

    def check_dir(self, dir):
        if not path.isdir(dir):
            assert not path.exists(dir), "Folder: " + str(dir) + " already exists as non-directory"
            mkpath(dir)

    def get_filename(self, dbname, filename):
        self.check_dir(self.folder)
        if(dbname):
            self.check_dir(self.folder + "/" + dbname)
            return self.folder + "/" + dbname + "/" + filename
        else:
            return self.folder + "/" + filename

    def download(self, url, file_name):
        u = urllib2.urlopen(url)
        f = open(file_name, 'w')
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        print "Downloading: %s Bytes: %s" % (file_name, file_size)

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += block_sz
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status)+1)
            print status,

        f.close()

def is_gzipfile(filename):
    try:
        gzip.open(filename,'r')
        gzip.close()
        return True
    except Exception:
        return False


def Unpack(filename, subfiles=None, force=False):
    p = path.dirname(filename)
    unpackpath = path.join(p, '/' + filename + '_unpacked/')
    if not path.isdir(unpackpath):
        os.mkdir(unpackpath)
    osubfiles = subfiles
    if isinstance(subfiles, str):
        subfiles = [subfiles]
    res = []
    statinfo = os.stat(filename)
    if zipfile.is_zipfile(filename):
        z = zipfile.ZipFile(filename,'r')
        if(not subfiles):
            subfiles = z.namelist()
        for subfile in subfiles:
            nspath = path.join(unpackpath, subfile)
            if not path.exists(nspath):
                z.extract(subfile, unpackpath)
            elif statinfo.st_ctime > os.stat(nspath).st_ctime:
                assert force, "Unpacking would overwrite older version. Delete the old data (in " + str(unpackpath) + ") or set force=True to proceed"
                os.unlink(nspath)
            res.append(nspath)

    elif tarfile.is_tarfile(filename):
        z = tarfile.open(filename, 'r')
        if(not subfiles):
            subfiles = z.getnames()
        for subfile in subfiles:
            nspath = path.join(unpackpath, subfile)
            if not path.exists(nspath):
                z.extract(subfile, unpackpath)
            elif statinfo.st_ctime > os.stat(nspath).st_ctime:
                assert force, "Unpacking would overwrite older version. Delete the old data (in " + str(unpackpath) + ") or set force=True to proceed"
                os.unlink(nspath)
                z.extract(subfile, unpackpath)
            res.append(nspath)
    else:
        return filename

    if isinstance(osubfiles, str):
        return res[0]
    else:
        return res        
        
            
            

    


    

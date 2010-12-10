#!/usr/bin/env python

from distutils.core import setup, Extension
import distutils.sysconfig
from distutils.debug import DEBUG
#import config
import sys

_ConfigDefault = {
    "setup.numpy_path" : distutils.sysconfig.get_python_lib() + \
        "/numpy/core/include",
    "setup.db_name":                "ibidas",
    "setup.db_user":                "postgres",
    "setup.db_user_pw ":            "Put the postgres password here",
    "setup.db_host":                "127.0.0.1",
    "setup.db_port":                "5432",
    "setup.authenticated_mode":     "yes",
    "setup.setup_type":             "new"
}
# include_dir = "/usr/lib64/python2.6/site-packages/numpy/core/include"

#setupconfig = config.loadConfig('setup.ini', _ConfigDefault)

include_dir = distutils.sysconfig.get_python_lib() + "/numpy/core/include"
#include_dir = "/usr/lib64/python2.6/site-packages/numpy/core/include"
#include_dir = setupconfig['setup.numpy_path']

setup(name = 'Ibidas', 
    version = '0.1.0', 
    description = 'The Ibidas data accession system',
    author = 'The Collective',
    author_email = 'letsnot@notgmail.com',
    url = 'https://wiki.nbic.nl/index.php/Ibidas',
    #py_modules = ['ibidas','test','config'],
    #packages = ['container', ],
    ext_modules = [
        Extension(
    		'closure', 
    		['src/closure.c'],
    		extra_compile_args=["-Wall"]
	    ),
        Extension(
		    'multi_visitor', 
		    ['src/multi_visitor.c'],
		    extra_compile_args=["-Wall"]
	    ),
        Extension(
            'base_container', 
            ['src/base_container.c'],
            include_dirs=[include_dir],
            extra_compile_args=["-Wall"]
        ),
        Extension(
            'cutils', 
            ['src/SFMT.c','src/cutils.c'], 
            include_dirs=[include_dir],
            extra_compile_args=["-Wall", "-ggdb3", "-O3", "-msse2", "-DHAVE_SSE2","-DMEXP=19937"]
#            extra_compile_args=["-Wall", "-O3", "-msse2","-msse4.2", "-DHAVE_SSE2","-DMEXP=19937"]
        ),
     ],
     requires=['numpy','sqlalchemy','ipython'],
     provides=['ibidas'],
)

#if (setupconfig['setup.setup_type'] == "new"):
#    args = []
#    
#    debug_answer = ""
#    while not(debug_answer == "y" or debug_answer == "n"):
#        debug_answer = raw_input("Do you want to run a debugging setup? (y/n)\n")
#    if debug_answer == "y":
#        args.append("debug")
#    
#    override_answer = ""
#    while not(override_answer == "y" or override_answer == "n"):
#        override_answer = raw_input("Do you want to override the database if it already exists? (y/n)\n")
#    if override_answer == "y":
#        args.append("override")
#    
#    soap_answer = ""
#    while not(soap_answer == "y" or soap_answer == "n"):
#        soap_answer = raw_input("Do you want to generate SOAP Serializers? (y/n)\n")
#    if soap_answer == "n":
#        args.append("nosoap")
#    
#    import os
#    if len(args) > 0:
#        add = " " + " ".join(args)
#    else:
#        add = ""
#    os.system("python setup_db.py" + add)

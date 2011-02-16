#!/usr/bin/env python
import ez_setup
ez_setup.use_setuptools()

from setuptools import setup,find_packages,Extension
import distutils.sysconfig
import os
import os.path

include_dir = distutils.sysconfig.get_python_lib() + "/numpy/core/include/"
if(not os.path.isfile(os.path.join(include_dir, "numpy/arrayobject.h"))):
    #print os.path.join(include_dir, "numpy/arrayobject.h")
    include_dir = include_dir.replace('lib', 'lib64')
    if(not os.path.isfile(os.path.join(include_dir, "numpy/arrayobject.h"))):
        #print os.path.join(include_dir, "numpy/arrayobject.h")
        raise RuntimeError, 'numpy array headers not found'

if not os.path.isdir('docs/_build'):
    os.mkdir('docs/_build')

setup(
    name="Ibidas",
    version="0.1.0",
    packages = find_packages(),
    test_suite = "test",
    scripts = ['bin/ibidas'],
    ext_modules = [
        Extension(
    		'closure', 
    		['src/closure.c'],
    		extra_compile_args=["-Wall"]
	    ),
        Extension(
		    'ibidas.utils.multi_visitor', 
		    ['src/multi_visitor.c'],
		    extra_compile_args=["-Wall"]
	    ),
        Extension(
            'ibidas.utils.cutils', 
            ['src/cutils.c'], 
            include_dirs=[include_dir],
            extra_compile_args=["-Wall"]
        ),
     ],
     install_requires=['numpy>=1.5.1','sqlalchemy>=0.6.4','ipython>=0.10.1','sphinx>=1.0.5'],
     author = "M. Hulsman & J.J.Bot",
     author_email = "m.hulsman@tudelft.nl",
     description = "Ibidas is an environment for data handling and exploration, able to cope with different data structures and sources",
     url = "https://trac.nbic.nl/ibidas/",
     license = "LGPLv2.1",

)


#!/usr/bin/env python
import ez_setup
ez_setup.use_setuptools()

from setuptools import setup,find_packages,Extension
import distutils.sysconfig
import os.path

include_dir = distutils.sysconfig.get_python_lib() + "/numpy/core/include/"
if(not os.path.isfile(os.path.join(include_dir, "numpy/arrayobject.h"))):
    #print os.path.join(include_dir, "numpy/arrayobject.h")
    include_dir = include_dir.replace('lib', 'lib64')
    if(not os.path.isfile(os.path.join(include_dir, "numpy/arrayobject.h"))):
        #print os.path.join(include_dir, "numpy/arrayobject.h")
        raise RuntimeError, 'numpy array headers not found'

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
            'ibidas.utils.base_container', 
            ['src/base_container.c'],
            include_dirs=[include_dir],
            extra_compile_args=["-Wall"]
        ),
        Extension(
            'ibidas.utils.cutils', 
            ['src/SFMT.c','src/cutils.c'], 
            include_dirs=[include_dir],
            extra_compile_args=["-Wall", "-ggdb3", "-O3", "-msse2", "-DHAVE_SSE2","-DMEXP=19937"]
#            extra_compile_args=["-Wall", "-O3", "-msse2","-msse4.2", "-DHAVE_SSE2","-DMEXP=19937"]
        ),
     ],
     install_requires=['numpy','sqlalchemy','ipython'],


)


#include <Python.h>
#include <math.h>
#include "numpy/arrayobject.h"
#define L_UNKNOWN -1

static PyObject * transpose(PyObject *self,PyObject *obj)
{
    int i;
    Py_ssize_t nrow;
    Py_ssize_t ncol = L_UNKNOWN, currow = 0;
    PyObject * res = NULL, * tmp;
    PyObject * iter = NULL;
    PyObject *item = NULL;

    if(PyArray_Check(obj))
    {
        Py_INCREF(obj);
        return obj;
    }

    if(!PySequence_Check(obj))
    {   
        PyErr_SetString(PyExc_TypeError,"argument should be a sequence");
        return NULL;
    }
    nrow = PySequence_Length(obj);
    if(nrow == L_UNKNOWN)
        return NULL;

    if(nrow == 0)
    {
        Py_INCREF(obj);
        return obj;
    }
    
    iter = PyObject_GetIter(obj);
    if(iter == NULL || PyErr_Occurred() || !PyIter_Check(iter))
        return NULL;
    
    while((item = PyIter_Next(iter)))
    {
        if(ncol == L_UNKNOWN) //initialization
        {
            if(!PySequence_Check(item))
            {
                PyErr_SetString(PyExc_TypeError,"argument should be a nested sequence");
                goto error;
            }
            ncol = PySequence_Length(item);
            if(ncol == L_UNKNOWN)
                goto error;

            res = PyTuple_New(ncol);
            if(res == NULL)
                goto error;
            
            for(i = 0; i < ncol; i++)
            {
                tmp = PyTuple_New(nrow);
                if(tmp == NULL)
                    goto error;
                PyTuple_SET_ITEM(res,i,tmp);
            }
        }
       
        /* Special casing for speed */
        if(PyTuple_Check(item))
        {
            if(PyTuple_GET_SIZE(item) != ncol)
            {
                PyErr_SetString(PyExc_TypeError,"Nested sequences should have equal length.");
                goto error;
            }
            for(i = 0; i < ncol; i++)
            {
                tmp = PyTuple_GET_ITEM(item,i);
                Py_INCREF(tmp);
                PyTuple_SET_ITEM(PyTuple_GET_ITEM(res,i),currow,tmp);
            }
        }
        else if(PyList_Check(item))
        {
            if(PyList_GET_SIZE(item) != ncol)
            {
                PyErr_SetString(PyExc_TypeError,"Nested sequences should have equal length.");
                goto error;
            }
            for(i = 0; i < ncol; i++)
            {
                tmp = PyList_GET_ITEM(item,i);
                Py_INCREF(tmp);
                PyTuple_SET_ITEM(PyTuple_GET_ITEM(res,i),currow,tmp);
            }
         }
        else if(PySequence_Check(item))
        {
            if(PySequence_Length(item) != ncol)
            {
                PyErr_SetString(PyExc_TypeError,"Nested sequences should have equal length.");
                goto error;
            }
            
            for(i = 0; i < ncol; i++)
            {
                tmp = PySequence_GetItem(item,i);
                if(tmp == NULL)
                    goto error;
                PyTuple_SET_ITEM(PyTuple_GET_ITEM(res,i),currow,tmp);
            }
        }
        else
        {
            PyErr_SetString(PyExc_TypeError,"argument should be a nested sequence");
            goto error;
        }
        currow++;

        Py_DECREF(item);
    }
    Py_DECREF(iter);
   
    return res;

error:
    Py_XDECREF(res);
    Py_XDECREF(item);
    Py_XDECREF(iter);
    return NULL;

}


//Array constructor function. Can be used to limit the dimensions
//of the resulting object. 
//dimarray(array data,dtype description,[max_dimensions] (default: 1), [min_dimensions] (default: 0))
static PyObject *
numpy_dimarray(PyObject *self, PyObject *args)
{
    PyObject * seq;
    PyArray_Descr * dtype = NULL;
    int max_dim = 1;
    int min_dim = 0;

    if(!PyArg_ParseTuple(args,"O|O&ii",&seq,PyArray_DescrConverter,(PyObject **) &dtype,&max_dim,&min_dim))
        return NULL;
   
    if(dtype == NULL)
    {
        if(!PyArray_DescrConverter((PyObject *) &PyBaseObject_Type, &dtype))
            return NULL;
    }
    
    //dtype has been increffed by PyArray_DescrConverter
    return (PyObject *) PyArray_FromAny(seq,dtype,min_dim,max_dim,0,NULL);
}



static PyMethodDef module_methods[] = {
    {"transpose", (PyCFunction) transpose, METH_O, "Transposes nested sequence"},
    {"darray", (PyCFunction) numpy_dimarray, METH_VARARGS, "Constructor for numpy arrays, with min/max dim support."},
    {NULL}  /* Sentinel */
};



#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initcutils(void) 
{
    PyObject* m;
    m = Py_InitModule3("cutils", module_methods,
                       "C utility functions");
    import_array();
    return;

}



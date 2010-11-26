#include <Python.h>

static PyObject * PyList_Copy(PyObject * lst)
{
    int i,len;
    PyObject * res, * elem;
    
    if(lst == NULL)
        return NULL;
    if(!PyList_Check(lst))
        return lst;
    
    len = PyList_GET_SIZE(lst);
    res = PyList_New(len);
    for(i = 0; i < len; i++)
    {
        elem = PyList_GET_ITEM(lst,i);
        Py_INCREF(elem);
        PyList_SET_ITEM(res,i,elem);
    }
    return res;
}

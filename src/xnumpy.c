#include <Python.h>
#include "structmember.h"

#include "numpy/arrayobject.h"

static PyObject * XArray_dim1(PyArrayObject *self, PyObject *op);

static PyTypeObject XArrayType;  //C type of the xnumpy extension type

typedef struct {
    PyArrayObject base;
    PyObject * colnames;         //col names to address the columns (e.g. obj.colname)
} XArrayObject;                  //C object represention for this type


static void
XArray_dealloc(XArrayObject* self)
{
    Py_XDECREF(self->colnames);

    //Call the arrayobject deallocator to handle it further
    ((PyArrayObject *)self)->ob_type->tp_base->tp_dealloc((PyObject *) self);
}

static PyObject *
XArray_alloc(PyTypeObject *type, Py_ssize_t nitems)
{
    PyObject *obj;
    
    /* nitems will always be 0 */
    obj = (PyObject *)PyArray_malloc(type->tp_basicsize);
    PyObject_Init(obj, type);
    ((XArrayObject *)obj)->colnames = NULL;
    return obj;
}

/* Xarray_item_slice overloads the obj[int:int] operator. Adds functionality
 * to copy colnames to the result if necessary. */
static PyObject *
XArray_slice(XArrayObject *self, Py_ssize_t ilow,
            Py_ssize_t ihigh)
{
    PyObject * res;
    //Call default Numpy implementation
    res = ((PyArrayObject *)self)->ob_type->tp_base->tp_as_sequence->sq_slice((PyObject *)self,ilow,ihigh);
    
    //If there are colnames, we have a result,
    //it is not a scalar and not equal to this object, and we are one-dimensional 
    if(self->colnames != NULL && res != NULL && PyArray_Check(res) && (PyObject *)self != res &&
        ((PyArrayObject *)self)->nd == 1 && res->ob_type == ((PyArrayObject *)self)->ob_type && ((XArrayObject*)res)->colnames != NULL)
    {
        //Apply slice function to colnames array
        Py_DECREF(((XArrayObject *)res)->colnames);
        ((XArrayObject *)res)->colnames = ((PyArrayObject*)self->colnames)->ob_type->tp_as_sequence->sq_slice((PyObject *)self->colnames,ilow,ihigh);

        //Check if it failed
        if(((XArrayObject *)res)->colnames == NULL)
        {
            Py_DECREF(res);
            return NULL;
        }
    }
    return res;
}


static PySequenceMethods XArray_sequence = {
    (lenfunc)NULL,                  /*sq_length*/
    (binaryfunc)NULL,               /*sq_concat is handled by nb_add*/
    (ssizeargfunc)NULL,
    (ssizeargfunc)NULL,             /*sq_item */
    (ssizessizeargfunc)XArray_slice,/*sq_slice */
    (ssizeobjargproc)NULL,          /*sq_ass_item*/
    (ssizessizeobjargproc)NULL,     /*sq_ass_slice*/
    (objobjproc) NULL,              /*sq_contains */
    (binaryfunc) NULL,              /*sg_inplace_concat */
    (ssizeargfunc)NULL,
};

/* Xarray_subscript_nice overloads the obj[op] operator. Adds functionality
 * to copy colnames to the result if necessary. */
static PyObject *
XArray_subscript_nice(XArrayObject *self, PyObject *op)
{
    PyObject * res;

    //Obtain result from numpy implementation
    res = ((PyArrayObject *)self)->ob_type->tp_base->tp_as_mapping->mp_subscript((PyObject *)self,op);
    //If there are colnames, we have a result, and it is not equal to this object,
    //and it is not a scalar type
    if(self->colnames != NULL && res != NULL && (PyObject *)self != res \
        && res->ob_type == ((PyArrayObject *)self)->ob_type && ((XArrayObject *)res)->colnames != NULL)
    {
        //Cache number of dimensions of self
        long nd;
        nd = ((PyArrayObject *)self)->nd;
        //Operand can be a tuple (index in multiple dims)
        if(PyTuple_Check(op))
        {
           //Obtain number of elements in tuple
           long ts = PyTuple_GET_SIZE(op);
           int i = -1;
           
           //if number of operand parts not equal to dimensions
           if(ts < nd)
               //search for ellips operator
               while(PyTuple_GET_ITEM(op,i) != Py_Ellipsis && i < ts) i++;

           //If equal to dimensions, or if we find an ellips (which should not be the last element)
           if(nd == ts || (i >= 0 && i < (ts-1)))
           {
                //get last part (dimension on which the colnames apply)
                op = PyTuple_GET_ITEM(op,ts-1);

                /* FIXME: should check more carefully if this op_part reduces its dimension */
                if(!(PyInt_Check(op) || PyLong_Check(op) || PyArray_IsScalar(op, Integer)))
                {
                    //Apply same selector also to colnames
                    Py_DECREF(((XArrayObject *)res)->colnames);
                    ((XArrayObject *)res)->colnames = XArray_dim1(((PyArrayObject*)self->colnames),op);
                    //((XArrayObject *)res)->colnames = ((PyArrayObject*)self->colnames)->ob_type->tp_as_mapping->mp_subscript((PyObject *)self->colnames,op);
                }
                else
                {
                    Py_DECREF(((XArrayObject *)res)->colnames);
                    ((XArrayObject *)res)->colnames = NULL;
                    return res; //circumvent shared check
                }


           }
        }
        //Operand is not a tuple
        //If this object is 1-dimensional, operand affects colnames
        else if(nd == 1)
        {
            Py_DECREF(((XArrayObject *)res)->colnames);
            ((XArrayObject *)res)->colnames = XArray_dim1(((PyArrayObject*)self->colnames),op);
            //((XArrayObject *)res)->colnames = ((PyArrayObject*)self->colnames)->ob_type->tp_as_mapping->mp_subscript((PyObject *)self->colnames,op);
        }
        
        //Shared check if colnames selection did not fail
        if(((XArrayObject *)res)->colnames == NULL)
        {
            Py_DECREF(res);
            return NULL;
        }
    }
    return res;
    
}

static PyMappingMethods XArray_mapping = {
    (lenfunc)0,              /*mp_length*/
    (binaryfunc)XArray_subscript_nice,       /*mp_subscript*/
    (objobjargproc)0,       /*mp_ass_subscript*/
};



//Some status indexes used in get/set functions
#define COLNOTFOUND 1
#define COLFOUND 2
#define MULTICOLFOUND 3

/* Overload set function, to enable setting columns using 
 * obj.colname = foo syntax */
static int
XArray_setattro(XArrayObject *self, PyObject *name, PyObject * value)
{

    int res;

    /* First see if numpy already implements it (e.g. for shape) */
    res = PyObject_GenericSetAttr((PyObject *) self,name,value);

    /* If not, and we have colnames, dimensions is larger than 0, we 
     * had an AttributeError an not another exception */
    if(res == -1 && self->colnames != NULL && ((PyArrayObject *)self)->nd != 0 && \
        PyErr_ExceptionMatches(PyExc_AttributeError))
    {
        /* First task: find name in colnames array.
         * Store indexes in a list. */
        long i,cmp;
        char status;
        status = COLNOTFOUND;
        //Store indexes of objects in result. 
        PyObject * result=NULL;

        PyObject *tmp, *val, *item;
        Py_ssize_t cols = PySequence_Length(self->colnames);

        //Store exception to restore it if we fail
        PyObject *ptype,*pvalue,*ptraceback;
        PyErr_Fetch(&ptype,&pvalue,&ptraceback);

        for(i = 0; i<cols; i++)
        {
           item = PySequence_GetItem(self->colnames,i);
           if(item == NULL)
                return -1;
           //Compare names
           cmp = PyObject_Compare(item,name);
           Py_DECREF(item);

           if(PyErr_Occurred())
              return -1; 
           
           //If names are equal
           if(cmp == 0)
           {
              //get Python object representation of index
              val = PyInt_FromLong(i);
              if(val == NULL)
                return -1;
              
              switch(status)
              {
                case COLNOTFOUND:
                    status = COLFOUND;
                    result = val;
                    break;
                case COLFOUND:
                    //An earlier match was already found.
                    //Need to create a list of indexes.
                    tmp = PyList_New(2);
                    if(tmp == NULL)
                        return -1;
                    PyList_SET_ITEM(tmp,0,result);
                    PyList_SET_ITEM(tmp,1,val);
                    result = tmp;
                    status = MULTICOLFOUND;
                    break;
                case MULTICOLFOUND:
                    PyList_Append(result,val);
                    Py_DECREF(val);
                    break;
                    
              }
           }
        }
        //If matching column names were found
        if(status != COLNOTFOUND)
        {
            PyObject *tmp;
            //If multiple dimensional, we need to include an ellipsis object
            if(((PyArrayObject *)self)->nd > 1)
            {
                tmp = Py_BuildValue("(ON)",Py_Ellipsis,result);
                if(tmp == NULL)
                    return -1;
                Py_DECREF(result);
                result = tmp;
            }
            res = ((objobjargproc)((PyArrayObject *)self)->ob_type->tp_base->tp_as_mapping->mp_ass_subscript)((PyObject *)self,result,value);
            Py_DECREF(result);
        }
        else
            PyErr_Restore(ptype,pvalue,ptraceback);
    }
    return res;
}


/* Overload get attribute function, to enable getting columns using 
 * obj.colname syntax */
static PyObject *
XArray_getattro(XArrayObject *self, PyObject *name)
{

    PyObject * res = NULL;
    /* First see if numpy already implements it (e.g. for shape) */
    res = PyObject_GenericGetAttr((PyObject *) self,name);
    


    /* If not, and we have colnames, dimensions is larger than 0, we 
     * had an AttributeError an not another exception */
    if(res == NULL && self->colnames != NULL && ((PyArrayObject *)self)->nd != 0 && \
        PyErr_ExceptionMatches(PyExc_AttributeError))
    {
        /* First task: find name in colnames array.
         * Store indexes in a list. */
        long i,cmp;
        char status;
        status = COLNOTFOUND;
        //Store indexes of objects in result. 
        PyObject * result=NULL;

        PyObject *tmp, *val, *item;
        Py_ssize_t cols = PySequence_Length(self->colnames);

        //Store exception to restore it if we fail
        PyObject *ptype,*pvalue,*ptraceback;
        PyErr_Fetch(&ptype,&pvalue,&ptraceback);
        
        for(i = 0; i<cols; i++)
        {
           item = PySequence_GetItem(self->colnames,i);
           if(item == NULL)
                return NULL;
           //Compare names
           cmp = PyObject_Compare(item,name);
           Py_DECREF(item);

           if(PyErr_Occurred())
              return NULL; 
           
           //If names are equal
           if(cmp == 0)
           {
              //get Python object representation of index
              val = PyInt_FromLong(i);
              if(val == NULL)
                return NULL;
              
              switch(status)
              {
                case COLNOTFOUND:
                    status = COLFOUND;
                    result = val;
                    break;
                case COLFOUND:
                    //An earlier match was already found.
                    //Need to create a list of indexes.
                    tmp = PyList_New(2);
                    if(tmp == NULL)
                        return NULL;
                    PyList_SET_ITEM(tmp,0,result);
                    PyList_SET_ITEM(tmp,1,val);
                    result = tmp;
                    status = MULTICOLFOUND;
                    break;
                case MULTICOLFOUND:
                    PyList_Append(result,val);
                    Py_DECREF(val);
                    break;
                    
              }
           }
        }
        //If matching column names were found
        if(status != COLNOTFOUND)
        {
            PyObject *tmp;
            //If multiple dimensional, we need to include an ellipsis object
            if(((PyArrayObject *)self)->nd > 1)
            {
                tmp = Py_BuildValue("(ON)",Py_Ellipsis,result);
                if(tmp == NULL)
                    return NULL;
                result = tmp;
            }
            res = ((binaryfunc)((PyArrayObject *)self)->ob_type->tp_as_mapping->mp_subscript)((PyObject *)self,result);
            Py_DECREF(result);
        }
        else
            PyErr_Restore(ptype,pvalue,ptraceback);
    }
    return res;
}




/* Implementation of the obj[op] operator, which always returns at least an 1-dimensional array */
static PyObject * XArray_dim1(PyArrayObject *self, PyObject *op)
{
    PyObject * ref;
    Py_intptr_t idx;
    Py_intptr_t dim0;
    PyObject * to_decref = NULL;


    //Shortcut for length-1 op's if dimension is 1
    if(self->nd == 1)
    {
        //Convert 1-length tuples to element in tuple
        if(PyTuple_Check(op) && PyTuple_GET_SIZE(op) == 1)
        {
            op = PyTuple_GET_ITEM(op,0);
        }
    
        //Check if operand is a single scalar. If so, go to simple section

        if(PyArray_Check(op) && PyArray_ISINTEGER(op) && 
            ((PyArrayObject *)op)->nd == 0)
        {
            op = PyArray_ToScalar(((PyArrayObject *)op)->data,op);
            if(op == NULL)
                return NULL;
            to_decref = op;
        }

        if (PyInt_Check(op))
        {
            idx = (Py_intptr_t) PyInt_AS_LONG(op);
            goto simple;
        }
        if(PyLong_Check(op) )
        {
            idx = (Py_intptr_t) PyLong_AsLong(op);        
            goto simple;
        }
        if(PyArray_IsScalar(op, Integer))
        {
            op = PyNumber_Long(op);
            if(op != NULL)
            {
                idx = (Py_intptr_t) PyLong_AsLong(op);        
                Py_XDECREF(op);
                goto simple;
            }         
        }
        /* FIXME: somehow this does not work
        * when PyNumberIndex gets an int it complains that it cannot convert a tuple???
        * 
        if (PyIndex_Check(op) && !PySequence_Check(op)) {
            PyObject* op = PyNumber_Index(op);
            if(op == NULL)
                return NULL;
            if (PyInt_Check(op))
            {
                idx = (Py_intptr_t) PyInt_AS_LONG(op);
                goto simple;
            }
            if(PyLong_Check(op) )
            {
                idx = (Py_intptr_t) PyLong_AsLong(op);        
                goto simple;
            }
        }
        */
    }
   
    ref = ((binaryfunc)self->ob_type->tp_as_mapping->mp_subscript)((PyObject *)self,op);
    Py_XDECREF(to_decref);

    /* If scalar convert back to array (expensive...) */
    if(ref && PyArray_IsAnyScalar(ref))
    {
        PyArrayObject * ref2;
        Py_intptr_t dims[1] = {1};
        PyArray_Descr *d = self->descr;
       
        //First construct a numpy array
        Py_INCREF(d);
        ref2= (PyArrayObject *)PyArray_FromAny((PyObject *)ref,d,0,1,0,NULL);
        Py_DECREF(ref);
        if(!ref2)
            return NULL;

        //Now make it an XArrayType
        Py_INCREF(d);
        ref = PyArray_NewFromDescr(&XArrayType,d,1,dims,0,ref2->data,ref2->flags,(PyObject *)ref2);
        ((PyArrayObject * )ref)->base = (PyObject *)ref2;
    }
    return ref;

simple:
    Py_XDECREF(to_decref);
    if(idx == -1 && PyErr_Occurred())
        return NULL;
    
    dim0 = self->dimensions[0];

    //Handle negative indexes
    if(idx < 0)
        idx += dim0;

    //Use trick to obtain always array result by using slice function with i:i+1 as operand
    if(idx >= 0 && idx < dim0)
        return self->ob_type->tp_as_sequence->sq_slice((PyObject *)self,idx,idx+1);
    else
    {
        PyErr_SetString(PyExc_IndexError,"index out of bounds");
        return NULL;
    }
}



/* Implementation of the obj[op] operator, which always returns at least an 2-dimensional array */
static PyObject * XArray_dim2(PyArrayObject *self, PyObject *op)
{
    PyObject *res;

    res = XArray_dim1(self,op);

    if(res != NULL && (((PyArrayObject *) res)->nd == 1))
    {
        PyArray_Dims nshape;
        PyObject *tmp;
        int newdim[2];
        nshape.ptr = newdim;
        nshape.len = 2;

        if(PyTuple_Check(op))
        {
            int s = PyTuple_GET_SIZE(op);
            PyObject *item = PyTuple_GET_ITEM(op,s-1);
             if((PyArray_Check(op) && PyArray_ISINTEGER(op) && ((PyArrayObject *)op)->nd == 0) ||
                 PyInt_Check(item) || PyLong_Check(item) || PyArray_IsScalar(item,Integer))
            {
                nshape.ptr[0] = ((PyArrayObject *)res)->dimensions[0];
                nshape.ptr[1] = 1;
            }
            else
            {
                nshape.ptr[1] = ((PyArrayObject *)res)->dimensions[0];
                nshape.ptr[0] = 1;
            }
        }
        else // row has been selected
        {
            nshape.ptr[1] = ((PyArrayObject *)res)->dimensions[0];
            nshape.ptr[0] = 1;
        }
        tmp = PyArray_Newshape((PyArrayObject *)res,&nshape,PyArray_ANYORDER);
        /* tmp should be an XArrayObject */
        Py_XINCREF(((XArrayObject *)res)->colnames);
        ((XArrayObject *)tmp)->colnames = ((XArrayObject *)res)->colnames;

        Py_DECREF(res);
        res = tmp;
    }

    return res;
}


static PyMethodDef XArray_methods[] = {
    {"dim1", (PyCFunction)XArray_dim1, METH_O,
     "Returns an array subselection"
    },
    {"dim2", (PyCFunction)XArray_dim2, METH_O,
     "Returns an array subselection"
    },
    {NULL} 
};



/* Obtain colunm names. If not there, return None */
static PyObject *
XArray_getColNames(XArrayObject *self, void *closure)
{
    if(self->colnames == NULL)
    {
        Py_INCREF(Py_None);
        return Py_None;
    }  
    else
    {
        Py_INCREF(self->colnames);
        return self->colnames;
    }
}

/* Set column names. Should be 1-dim numpy array */
static int
XArray_setColNames(XArrayObject *self, PyObject *value, void *closure)
{
    if(!PyArray_Check(value) || ((PyArrayObject*)value)->nd != 1)
    {
        PyErr_SetString(PyExc_TypeError, "Column names should be given as a 1-dimensional numpy array");
        return -1;
    }
    Py_INCREF(value);
    self->colnames = value;
    return 0;
}

/* Obtain __array_priority__ */
static PyObject *
XArray_getPriority(XArrayObject *self, void *closure)
{
    return PyFloat_FromDouble(1.0);
}

static int XArray_Finalize(PyArrayObject * self, PyObject *parent)
{
    if(parent != NULL && parent->ob_type == &XArrayType && ((XArrayObject *)parent)->colnames != NULL)
    {
        Py_INCREF(((XArrayObject *)parent)->colnames);
        ((XArrayObject *)self)->colnames = ((XArrayObject*)parent)->colnames;
    }
    return 0;
}

static PyObject *
XArray_getArrayFinalize(XArrayObject *self, void *closure)
{
    return PyCObject_FromVoidPtr((void *)XArray_Finalize,NULL);
}

static PyGetSetDef XArray_getseters[] = {
    {"colnames",
     (getter)XArray_getColNames,(setter)XArray_setColNames,
     "This r/w attribute stores the column names",
     NULL},
    {"__array_priority__",
     (getter)XArray_getPriority,NULL,
     "Obtain __array_priority__ (read-only)",
     NULL},
    {"__array_finalize__",
     (getter)XArray_getArrayFinalize,NULL,
     "Obtain __array_finalize__ cobject(read-only)",
     NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject XArrayType = {
    PyObject_HEAD_INIT(NULL)
    0,                              /*ob_size*/
    "xnumpy.XNumpy",                /*tp_name*/
    sizeof(XArrayObject),           /*tp_basicsize*/
    0,                              /*tp_itemsize*/
    (destructor)XArray_dealloc,     /*tp_dealloc*/
    0,                              /*tp_print*/
    0,                              /*tp_getattr*/
    0,                              /*tp_setattr*/
    0,                              /*tp_compare*/
    0,                              /*tp_repr*/
    0,                              /*tp_as_number*/
    0,                              /*tp_as_sequence*/
    0,                              /*tp_as_mapping*/
    0,                              /*tp_hash */
    0,                              /*tp_call*/
    0,                              /*tp_str*/
    (getattrofunc)XArray_getattro,  /*tp_getattro*/
    (setattrofunc)XArray_setattro,  /*tp_setattro*/
    0,                              /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES, /*tp_flags*/
    "XArray object",                /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,                     /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    XArray_methods,        /* tp_methods */
    0,                     /* tp_members */
    XArray_getseters,      /* tp_getset */
    0,                     /* tp_base */
    0,                     /* tp_dict */
    0,                     /* tp_descr_get */
    0,                     /* tp_descr_set */
    0,                     /* tp_dictoffset */
    0,                     /* tp_init */
    XArray_alloc,          /* tp_alloc */
    0,                     /* tp_new */
};



//Array constructor function. Can be used to limit the dimensions
//of the resulting object. 
//dimarray(array data,dtype description,[max_dimensions] (default: 1), [min_dimensions] (default: 0))
static PyObject *
xnumpy_dimarray(PyObject *self, PyObject *args)
{
    PyObject * seq;
    PyArrayObject * new_array;
    PyObject * new_array2;
    PyArray_Descr * dtype = NULL;
    int max_dim = 1;
    int min_dim = 0;

    if(!PyArg_ParseTuple(args,"OO&|ii",&seq,PyArray_DescrConverter,(PyObject **) &dtype,&max_dim,&min_dim))
        return NULL;
   
    if(dtype == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Second argument should be an numpy dtype");
        return NULL;
    }
    

    new_array = (PyArrayObject *) PyArray_FromAny(seq,dtype,min_dim,max_dim,0,NULL);
    
    /*SIGH, we cannot create a subtype directly, so have to convert the new array to our subtype...*/
    if(new_array)
    {
        dtype = new_array->descr;
        Py_INCREF(dtype);
        new_array2 = PyArray_NewFromDescr(&XArrayType,dtype,new_array->nd,new_array->dimensions,new_array->strides,new_array->data,new_array->flags,(PyObject *)new_array);
        if(new_array2 == NULL)
            return NULL;
        ((PyArrayObject * )new_array2)->base = (PyObject *) new_array;
        return new_array2;
    }
    return NULL;
}


static PyMethodDef module_methods[] = {
    {"dimarray", (PyCFunction) xnumpy_dimarray, METH_VARARGS, "Creates a column array from a sequence"},
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initxnumpy(void) 
{
    PyObject* m;
    m = Py_InitModule3("xnumpy", module_methods,
                       "Numpy extension array that always returns arays.");
    import_array();

    XArrayType.tp_base = (struct _typeobject *) &PyArray_Type;
    XArrayType.tp_as_sequence = &XArray_sequence;    
    XArrayType.tp_as_mapping = &XArray_mapping;    


    if (PyType_Ready(&XArrayType) < 0)
        return;

    if (m == NULL)
      return;

    Py_INCREF(&XArrayType);
    PyModule_AddObject(m, "XArray", (PyObject *)&XArrayType);
}

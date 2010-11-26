#include "itypes.h"
#include "base_container.h"
#include <math.h>


static PyTypeObject ITypeType;

typedef struct {
    PyObject_HEAD
    it_code typecode;
    PyObject * typelength;
    PyObject * subtype;
} IType;

static void
IType_dealloc(IType* self)
{
    Py_XDECREF(self->subtype);
    self->ob_type->tp_free((PyObject*)self);
}


/* Creates new type. If a certain type is already in the cache that one is returned instead 
 * Typelength: NULL if scalar
 *             int if dim1
 *             tuple with ints if dim2
 * Subtype:    NULL if scalar, else subtype object
 */

static PyObject * _itypenew(PyTypeObject * cls,it_code typecode,PyObject * typelength,PyObject *subtype)
{
    IType *self = NULL;
    PyObject * cache, *key = NULL;
    int i;

    cache = cls->tp_dict;
    
    if(!VALID_TYPE(typecode))
    {
        PyErr_SetString(PyExc_TypeError,"TypeCode does not contain a valid typecode.");
        goto error;
    }
    

    /* Fill in typelength, subtype for str, unicode, bytes type if not already done so by
     * caller */
    if(IS_TYPE(typecode,STR | UNICODE | BYTES))
    {
        it_code stcode;
        PyObject *tmp;
        switch(typecode)
        {
            case STR:
                stcode = UINT8;
                break;
            case UNICODE:
                if(sizeof(Py_UNICODE) == 2)
                    stcode = UINT16;
                else
                    stcode = UINT32;
                break;
            case BYTES:
                stcode = UINT8;
                break;
        }
        tmp = _itypenew(cls,stcode,NULL,NULL);

        if(subtype != NULL && subtype != tmp)
        {
            PyErr_SetString(PyExc_TypeError,"Subtype contains invalid object");
            goto error;
        }
        else
            subtype = tmp;

        if(typelength != NULL)
        {
            if(!PyInt_Check(typelength))
            {
                PyErr_SetString(PyExc_TypeError,"Invalid length set");
                goto error;
            }
        }
        else
            typelength = PyInt_FromLong(L_UNKNOWN);
    }

    if(typelength != NULL)
    {
        if(PyTuple_Check(typelength))
        {
            for(i = 0; i < PyTuple_GET_SIZE(typelength); i++)
            {
                if(!PyInt_Check(PyTuple_GET_ITEM(typelength,i)))
                {
                    PyErr_SetString(PyExc_TypeError,"Typelength tuple should contain integers.");
                    goto error;
                }
            }
        }
        else if(!PyInt_Check(typelength))
        {
            PyErr_SetString(PyExc_TypeError,"Typelength should be an integer or a tuple of integers.");
            goto error;
        }
    }
    
    
    if(!((IS_TYPE(typecode,SEQUENCE)  && typelength != NULL) || 
         (!IS_TYPE(typecode,SEQUENCE) && typelength == NULL)))
    {
        PyErr_SetString(PyExc_TypeError,"Invalid typelength set");
        goto error;
    }

    
    if(!((IS_TYPE(typecode,SEQUENCE)  && subtype != NULL) || 
         (!IS_TYPE(typecode,SEQUENCE) && subtype == NULL)))
    {
        PyErr_SetString(PyExc_TypeError,"Invalid subtype set.");
        goto error;
    }

    if(typelength == NULL)
        if(subtype == NULL)
            key = PyInt_FromLong(typecode);
        else
            key = Py_BuildValue("kl",typecode,(long) subtype);
    else
        key = Py_BuildValue("kOl",typecode,typelength,(long) subtype);

    if(key == NULL)
        goto error;

    self = (IType *) PyDict_GetItem(cache,key);
    if(self != NULL)
        Py_INCREF(self);
    else
    {
        self = (IType *)cls->tp_alloc(cls, 0);
        if(self == NULL)
            goto error;

        self->typecode = typecode;
        Py_XINCREF(typelength);
        self->typelength = typelength;
        Py_XINCREF(subtype);
        self->subtype = subtype;

        if(PyDict_SetItem(cache,key,(PyObject *) self) == -1)
            goto error;
    }
    Py_DECREF(key);
    return (PyObject *)self;
error:
    Py_XDECREF(key);
    Py_XDECREF(self);
    return NULL;
 
}

/* Main new function for IType calls from python */
static PyObject * IType_new(PyTypeObject * cls, PyObject *args, PyObject *kwds)
{
    it_code typecode;
    PyObject * typec;
    PyObject * typelength = NULL;
    PyObject * subtype = NULL, * tmp, * res;
    static char *kwlist[] = {"typecode", "typelength", "subtype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO", kwlist, 
                &typec, &typelength,&subtype))
        return NULL;
   
    /* Typecode can also be name, convert using dict lookup */
    if(!PyDict_Check(typedict))
    {
        PyErr_SetString(PyExc_TypeError,"The module type dict is not a dictionary anymore!");
        return NULL;
    }
    tmp = PyDict_GetItem(typedict,typec);
    /* replace type code if conversion found */
    if(tmp != NULL)
        typec = tmp;

    /* Check / convert typecode to long */
    if(PyInt_Check(typec))
        typecode = PyInt_AsLong(typec);
    else if(PyLong_Check(typec))
        typecode = PyLong_AsLong(typec);
    else
    {
        PyErr_SetString(PyExc_TypeError,"Invalid typecode!");
        return NULL;
    }
    if(PyErr_Occurred())  //Possible error in PYint/long to long conversion
        return NULL;
   
    /* Check typelength, ensure its a tuple (own incref after)*/
    if(typelength != NULL)
    {
        if(PyLong_Check(typelength))
        {
            tmp = PyInt_FromLong(PyLong_AsLong(typelength));
            Py_DECREF(typelength);
            typelength = tmp;
        }
        else if(!PyTuple_Check(typelength) && !PyInt_Check(typelength))
        {
            PyErr_SetString(PyExc_TypeError,"Incorrect typelength format (should be scalar or tuple)");
            return NULL;
        }
        else
            Py_INCREF(typelength);
    }

    res = _itypenew(cls,typecode,typelength,subtype);
    Py_XDECREF(typelength);
    return res;
}



static PyMethodDef IType_methods[] = {
    {NULL} 
};








static PyObject *
IType_gettypecode(IType *self, void *closure)
{
    PyObject * tmp = Py_typecodes[get_type_nr(self->typecode)];
    Py_XINCREF(tmp);
    return tmp;
}


static PyObject *
IType_gettypelength(IType *self, void *closure)
{
    if(self->typelength != NULL)
    {
        Py_INCREF(self->typelength);
        return self->typelength;
    }
    else
        Py_RETURN_NONE;
}

static PyObject *
IType_getsubtype(IType *self, void *closure)
{
    if(self->subtype != NULL)
    {
        Py_INCREF(self->subtype);
        return self->subtype;
    }
    else
        Py_RETURN_NONE;
}


static PyGetSetDef IType_getseters[] = {
    {"typecode", 
     (getter)IType_gettypecode, NULL,
     "typecode",
     NULL},
    {"typelength", 
     (getter)IType_gettypelength, NULL,
     "typelength",
     NULL},
    {"subtype", 
     (getter)IType_getsubtype, NULL,
     "subtype",
     NULL},
    {NULL}  /* Sentinel */
};


static PyObject *
IType_repr(IType *self)
{
   PyObject * res, *tmp;;
   int i,ll;
   long tl;

   res = PyString_FromString(type_names[get_type_nr(self->typecode)]);
   if(res == NULL)
        return NULL;

   /* somewhat inefficient ... */
   if(self->subtype != NULL)
   {
       tmp = PyObject_Repr(self->subtype);
       if(tmp == NULL)
           return NULL;
       PyString_ConcatAndDel(&res,PyString_FromFormat("(%s)",PyString_AS_STRING(tmp)));
       if(res == NULL)
            return NULL;
   }

   if(self->typelength != NULL)
   {
       if(PyInt_Check(self->typelength))
       {
           tl = PyInt_AsLong(self->typelength);
           if(tl == L_UNKNOWN)
               PyString_ConcatAndDel(&res,PyString_FromString("[]"));
           else
               PyString_ConcatAndDel(&res,PyString_FromFormat("[%ld]",tl));
       }
       else
       {
           ll = PyTuple_GET_SIZE(self->typelength); 
           PyString_ConcatAndDel(&res,PyString_FromString("["));
           if(res == NULL)
               return NULL;
           for(i = 0; i < ll; i++)
           {
               tl = PyInt_AsLong(PyTuple_GET_ITEM(self->typelength,i));
               if(PyErr_Occurred())
                  return NULL;
               if(tl == L_UNKNOWN)
                   PyString_ConcatAndDel(&res,PyString_FromString(":"));
               else
                   PyString_ConcatAndDel(&res,PyString_FromFormat("%ld",tl));
               if(res == NULL)
                   return NULL;
               
               if(i < (ll -1))
                    PyString_ConcatAndDel(&res,PyString_FromString(","));
           }
           PyString_ConcatAndDel(&res,PyString_FromString("]"));
       }
   }
   return res;
}

static PyTypeObject ITypeType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "itypes.IType",             /*tp_name*/
    sizeof(IType),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)IType_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)IType_repr,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                          /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_CLASS, /*tp_flags*/
    "IType object",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,                     /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    IType_methods,         /* tp_methods */
    0,                         /* tp_members */
    IType_getseters,          /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                     /* tp_init */
    0,                         /* tp_alloc */
    IType_new,                 /* tp_new */
};

static PyObject *
IType_constructor(PyObject *self, PyObject * args)
{
    PyObject *name = NULL, *subtype = NULL, *tlength = NULL, *length = NULL;
    PyObject *tmp, *tmp2, *nargs = NULL;
    PyObject * res = NULL;
    int pcounter = 0;
    int state = 0;
    int i;

    if(!PyString_Check(args) && PySequence_Check(args))
       return _callback_wrapper(self,args,IType_constructor);

    if(PyString_Check(args))
    {
        char * descr;
        int pos = 0, oldpos = 0;

        descr = PyString_AsString(args);
        
        length = PyList_New(0);
        while(descr[pos])
        {
           switch(descr[pos])
           {
               case '(':
                    pcounter++;
                    if(state == STATE_START)
                    {
                        name =  PySequence_GetSlice(args,oldpos,pos);
                        if(name == NULL)
                            return NULL;
                        oldpos = pos;
                        state = STATE_INSUBTYPE;
                    }
                    else if(state != STATE_INSUBTYPE)
                    {
                        PyErr_SetString(PyExc_TypeError,"Unexpected ( encoutered");
                        goto error;
                    }
                    break;
               case ')':
                    if(state != STATE_INSUBTYPE)
                    {
                        PyErr_SetString(PyExc_TypeError,"Unexpected , or ) encoutered");
                        goto error;
                    }
                    else if(pcounter == 1)
                    {
                        tmp = PySequence_GetSlice(args,oldpos + 1,pos);
                        if(tmp == NULL) return NULL;
                        subtype = IType_constructor(self,tmp);
                        Py_DECREF(tmp);
                        if(subtype == NULL)
                            goto error;
                        oldpos = pos;
                        state = STATE_ENDSUBTYPE;
                    }
                    pcounter--;
                    break;
               case '[':
                        if(state == STATE_INSUBTYPE)
                            state = STATE_SUBTYPEINBLOCK;
                        else if(state == STATE_ENDSUBTYPE)
                        {
                            state = STATE_INBLOCK;
                            oldpos = pos;
                            pcounter++;
                        }
                        else
                        {
                            PyErr_SetString(PyExc_TypeError,"Unexpected [ encoutered");
                            goto error;
                        }
                        break;
               case ',':
               case ']':
                        if(state == STATE_SUBTYPEINBLOCK && descr[pos] == ']')
                            state = STATE_INSUBTYPE;
                        else if(state == STATE_INBLOCK)
                        {
                            if(oldpos + 1 < pos)
                            {
                                tmp = PySequence_GetSlice(args,oldpos+1,pos);
                                if(tmp == NULL)
                                    return NULL;
                                char * k = PyString_AsString(tmp);
                                tmp2 = PyInt_FromString(k,NULL,10);
                                Py_DECREF(tmp);
                            }
                            else
                                tmp2 = PyInt_FromLong(L_UNKNOWN);
                            if(tmp2 == NULL)
                                return NULL;
                            if(PyList_Append(length,tmp2) < 0)
                                return NULL;
                            Py_DECREF(tmp2);
                            oldpos = pos;
                            if(descr[pos] == ']')
                            {
                                pcounter--;
                                state = 4;
                            }
                        }
                        else
                        {
                            PyErr_SetString(PyExc_TypeError,"Unexpected ] encoutered");
                            goto error;
                        }
                        break;
           }
           pos ++;
        }

        if(pcounter > 0 || state == STATE_INSUBTYPE || state == STATE_INBLOCK)
        {
             PyErr_SetString(PyExc_TypeError,"Unexpected end of descriptor");
             goto error;
        }
        
        /* copy lengths from list to integer or tuple of integers */
        if(PyList_GET_SIZE(length) == 1)
        {
            tlength = PyList_GET_ITEM(length,0);
            Py_INCREF(tlength);
        }
        else if(PyList_GET_SIZE(length) != 0)
        {
            tlength = PyTuple_New(PyList_GET_SIZE(length));
            for(i = 0; i < PyList_GET_SIZE(length); i++)
            {
                tmp = PyList_GET_ITEM(length,i);
                Py_INCREF(tmp);
                PyTuple_SET_ITEM(tlength,i,PyList_GET_ITEM(length,i));
            }
        }

        /* create args for constructor call */
        if(state == STATE_START)
            nargs = PyTuple_Pack(1,args);
        else if(state == STATE_ENDSUBTYPE)
            nargs = Py_BuildValue("(NiN)",name,L_UNKNOWN,subtype);
        else if(state == STATE_ENDBLOCK)
            nargs = Py_BuildValue("(NNN)",name,tlength,subtype);
        
        if(nargs == NULL)
            goto error;
        
        res = IType_new(&ITypeType,nargs,NULL);
        

        Py_DECREF(nargs); //should decref name, tlength,subtype
        Py_DECREF(length);
    }
    else
    {
        PyObject *dtype;
        //lets try if numpy can do something with it
        if(PyArray_DescrConverter(args,(PyArray_Descr **)&dtype) == 0)
        {
            PyErr_SetString(PyExc_TypeError,"Unsupported type description");
            return NULL;
        }
        res = IType_from_numpy(self,dtype);
        Py_DECREF(dtype);
    }

    return res;
error:
    Py_XDECREF(name);
    Py_XDECREF(subtype);
    Py_XDECREF(tlength);
    Py_XDECREF(length);
    return NULL;
}

/* returns type of object in args */
static PyObject *
IType_getType(PyObject *self, PyObject *args)
{
    PyObject * pytc;
    it_code typecode;
    PyObject * ar_length = NULL;
    PyObject * subtype = NULL;
    PyObject * res;

    if(convdict == NULL)
        return NULL;
   
    /* obtain type code */
    pytc = register_pytype(args);
    if(pytc == NULL)
        return NULL;
    
    typecode = PyInt_AsLong(pytc);
    if(PyErr_Occurred())
        return NULL;

    /* get sequence length */
    if(IS_TYPE(typecode,SEQUENCE))
    {
        long tlength = PySequence_Length(args);
        ar_length = PyInt_FromLong(tlength);
        if(ar_length == NULL)
            return NULL;
    }
    
    /* get subtype */
    if(IS_TYPE(typecode,RSEQUENCE))
    {
       subtype = _getseqtype(self,args,typecode,1);
       if(subtype == NULL)
            return NULL;
    }
    
    res = _itypenew(&ITypeType,typecode,ar_length,subtype);
    Py_XDECREF(ar_length);
    Py_XDECREF(subtype);
    return res;
}


static PyObject * IType_getSeqType(PyObject * self, PyObject *args)
{
    PyObject * res;
    PyObject * res2;
    if(!PySequence_Check(args))
    {
        PyErr_SetString(PyExc_TypeError,"argument should be a sequence");
        return 0;
    }
    res = IType_getType(self,args);
    if(res == NULL)
        return NULL;

    res2 = ((IType *)res)->subtype;
    Py_XINCREF(res2);
    Py_DECREF(res);
    return res2;
}

static PyObject * IType_getSeqSeqType(PyObject * self, PyObject *args)
{
    PyObject * res, * tmp, *stype;
    IType *type;
    Py_ssize_t argslength;
    int i;

    if(!PySequence_Check(args))
    {
        PyErr_SetString(PyExc_TypeError,"argument should be a sequence");
        return 0;
    }
    argslength = PySequence_Length(args);
    if(argslength == L_UNKNOWN)
        return NULL;
    
    res = PyTuple_New(argslength);
    for(i = 0; i < argslength; i++)
    {
        tmp = PySequence_GetItem(args,i);
        if(tmp == NULL)
            return NULL;
        type = (IType *) IType_getType(self,tmp);
        if(type == NULL)
            return NULL;
        stype = type->subtype;
        Py_XINCREF(stype);
        Py_DECREF(type);
        if(stype != NULL)
            PyTuple_SET_ITEM(res,i,stype);
        else
        {
            
            PyErr_SetString(PyExc_TypeError,"Incorrect subtype.");
            return NULL;
        }
    }
    return res;
}


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
static PyObject * IType_to_defval(PyObject *self, PyObject *obj)
{
    if(PySequence_Check(obj))
        return _callback_wrapper(self,obj,IType_to_defval);
    if(!PyObject_IsInstance(obj,(PyObject *) &ITypeType))
    {
        PyErr_SetString(PyExc_TypeError,"Argument should be a IType");
        return NULL;
    }
    return get_defval(((IType *)obj)->typecode);
}


static PyObject * IType_to_numpy(PyObject *self, PyObject *obj)
{
    it_code typecode;
    PyObject *typelength;
    PyObject *name;
    PyObject *dtype;
    if(PySequence_Check(obj))
        return _callback_wrapper(self,obj,IType_to_numpy);


    if(!PyObject_IsInstance(obj,(PyObject *) &ITypeType))
    {
        PyErr_SetString(PyExc_TypeError,"Argument should be a IType");
        return NULL;
    }
    typecode = ((IType *)obj)->typecode;
    name = PyString_FromString(numpy_names[get_type_nr(typecode)]);
    if(name == NULL)
        return NULL;
    
    if(IS_TYPE(typecode,STR | UNICODE | BYTES))
    {
        typelength = ((IType *)obj)->typelength;
        long typelength_long = PyInt_AsLong(typelength);
        typelength_long = MAX(typelength_long,1);
        if(typelength == NULL || PyTuple_Check(typelength) || typelength_long > 12 || (PyInt_Check(typelength) && PyInt_AsLong(typelength) == L_UNKNOWN))
        {
            Py_DECREF(name);
            name = PyString_FromString(numpy_names[IOBJECT]);
        }
        else
            PyString_ConcatAndDel(&name,PyString_FromFormat("%ld",typelength_long));  
        if(name == NULL)
            return NULL;
    }
    
    if(PyArray_DescrConverter(name,(PyArray_Descr **)&dtype) == 0)
        return NULL;

    return dtype;
}


static PyObject * IType_from_numpy(PyObject *self, PyObject *obj)
{
    PyObject *res, *pytc; 
    it_code typecode;
    if(PySequence_Check(obj))
        return _callback_wrapper(self,obj,IType_from_numpy);
    if(PyArray_DescrCheck(obj))
    {
        pytc = _dtype_to_typecode(obj);
        if(pytc == NULL)
            return NULL;
        typecode = PyInt_AsLong(pytc);
        if(PyErr_Occurred())
            return NULL;

        if(IS_TYPE(typecode,~(SEQUENCE)))
            res = _itypenew(&ITypeType,typecode,NULL,NULL); 
        else
        {
            PyObject * itemsize;
            int isize, length;
            switch(typecode)
            {
                case STR:
                    isize = 1;
                    break;
                case UNICODE:
                    isize = sizeof(Py_UNICODE);  // i'm not sure that numpy uses the same size...
                    break;
                case BYTES:
                    isize = 1;
                    break;
                default:
                    PyErr_SetString(PyExc_TypeError,"Unexpected numpy type.");
                    return NULL;
            }
            
            itemsize = PyObject_GetAttrString(obj,"itemsize");
            if(itemsize == NULL)
            {
                PyErr_SetString(PyExc_TypeError,"Numpy scalar dtype has no itemsize.");
                Py_XDECREF(itemsize);
                return NULL;
            }
            
            length = PyInt_AsLong(itemsize) / isize;
            if(length < 0)
                return NULL;
            if(length == 0)
                length = L_UNKNOWN;
            res = _itypenew(&ITypeType,typecode,PyInt_FromLong(length),NULL);
            Py_DECREF(itemsize);
        }
    }
    else res= IType_getType(self,obj);

    return res;
}

static PyObject * _callback_wrapper(PyObject *self, PyObject *args, PyCFunction callback)
{
    if(!PySequence_Check(args))
    {
        PyErr_SetString(PyExc_TypeError,"Internal error in wrapper function. Argument should be sequence.");
        return NULL;
    }
    else
    {
        PyObject *res, *tmp, *tmp2;
        int i;
        Py_ssize_t length = PySequence_Length(args);
        if(length == L_UNKNOWN)
            return NULL;
        res = PyTuple_New(length);
        if(res == NULL)
            return NULL;
        for(i = 0; i < length; i++)
        {
            tmp = PySequence_GetItem(args,i);
            if(tmp == NULL)
                return NULL;
            tmp2 = callback(self,tmp);
            if(tmp2 == NULL)
                return NULL;
            Py_DECREF(tmp);
            PyTuple_SET_ITEM(res,i,tmp2);
        }
        return res;
    }
}





static PyObject * chooseType(PyObject *self, PyObject *typecodes)
{
    if(PyInt_Check(typecodes))
    {
        PyObject * res;
        long tcs = PyInt_AsLong(typecodes);
        if(PyErr_Occurred())
            return NULL;
        tcs = choose_type(tcs);
        res = PyInt_FromLong(tcs);
        return res;
    }

    PyErr_SetString(PyExc_TypeError,"Argument should be an integer");
    return NULL;
}


static PyObject * getConvertableTypes(PyObject * self, PyObject *typecode)
{
    if(PyInt_Check(typecode))
    {
        PyObject * res;
        long tcs = PyInt_AsLong(typecode);
        if(PyErr_Occurred())
            return NULL;
        tcs = get_convertable_types(tcs);
        res = PyInt_FromLong(tcs);
        return res;
    }

    PyErr_SetString(PyExc_TypeError,"Argument should be an integer");
    return NULL;
}

static PyObject * getCompatibleTypes(PyObject * self, PyObject *typecode)
{
    if(PyInt_Check(typecode))
    {
        PyObject * res;
        long tcs = PyInt_AsLong(typecode);
        if(PyErr_Occurred())
            return NULL;
        tcs = get_compatible_types(tcs);
        res = PyInt_FromLong(tcs);
        return res;
    }

    PyErr_SetString(PyExc_TypeError,"Argument should be an integer");
    return NULL;
}


static PyMethodDef module_methods[] = {
    {"getType", (PyCFunction) IType_getType, METH_O, "Retrieves types"},
    {"getSeqType", (PyCFunction) IType_getSeqType, METH_O, "Retrieves internal type of a sequence"},
    {"getSeqSeqType", (PyCFunction) IType_getSeqSeqType, METH_O, "Retrieves internal types of a nested sequence"},
    {"transpose", (PyCFunction) transpose, METH_O, "Transposes nested sequence"},
    {"createType", (PyCFunction) IType_constructor, METH_O, "Create new type from description"},
    {"from_numpy", (PyCFunction) IType_from_numpy, METH_O, "Numpy dtype to Type convertor"},
    {"to_numpy", (PyCFunction) IType_to_numpy, METH_O, "Type to numpy dtype convertor"},
    {"to_defval", (PyCFunction) IType_to_defval, METH_O, "Return default value for this type."},
    {"chooseType", (PyCFunction) chooseType, METH_O, "Returns most specific/smallest type."},
    {"getConvertableTypes", (PyCFunction) getConvertableTypes, METH_O, "Returns the types a specific type can be converted to."},
    {"getCompatibleTypes", (PyCFunction) getCompatibleTypes, METH_O, "Returns the types a specific type can be converted to without loss."},
    {NULL}  /* Sentinel */
};


/* combine two type descriptions, finds common supertype, otherwise returns NULL */
static PyObject * _common_subtype(IType * args1,IType * args2)
{
    PyObject * res;
    PyObject * tmp1, * tmp2;
    int i;
    it_code ntc;
    if(args1 == NULL || args2 == NULL)
    {
        if(args1 == NULL && args2 != NULL)
        {
            Py_INCREF(args2);
            return (PyObject *) args2;
        }
        else if(args1 != NULL && args2 == NULL)
        {
            Py_INCREF(args1);
            return (PyObject *) args1;
        }
        else return NULL;
    }
  
    if(!PyObject_IsInstance((PyObject *)args1,(PyObject *)&ITypeType) || 
       !PyObject_IsInstance((PyObject *)args2,(PyObject *)&ITypeType))
        return NULL;

    if(args1 == args2)
    {
        Py_INCREF(args1);
        res = (PyObject *) args1;
    }
    ntc = choose_type(get_compatible_types(args1->typecode) & get_compatible_types(args2->typecode));
    if(IS_TYPE(ntc,SEQUENCE))
    {
        PyObject * length;
        if(args1->typelength == NULL && args2->typelength == NULL)
            length = NULL;
        else if(PyInt_Check(args1->typelength) && PyInt_Check(args2->typelength))
        {
           length = (PyObject *) ((PyInt_AsLong((PyObject *)args1->typelength) > PyInt_AsLong((PyObject *)args2->typelength)) ? args1->typelength : args2->typelength);
           Py_INCREF(length);
        }
        else if(PyTuple_Check(args1->typelength) && PyTuple_Check(args2->typelength) && PyTuple_GET_SIZE(args1->typelength) == PyTuple_GET_SIZE(args2->typelength))
        {
            length = PyTuple_New(PyTuple_GET_SIZE(args1->typelength));
            if(length == NULL)
                return NULL;
            for(i = 0; i < PyTuple_GET_SIZE(args1->typelength); i++)
            {
                tmp1 = PyTuple_GET_ITEM(args1->typelength,i);
                tmp2 = PyTuple_GET_ITEM(args2->typelength,i);
                tmp1 = ((PyInt_AsLong(tmp1) > PyInt_AsLong(tmp2)) ? tmp1 : tmp2);
                Py_INCREF(tmp1);
                PyTuple_SET_ITEM(length,i,tmp1);
            }
        }
        else
        {
            ntc = OBJECT;
            length = NULL;
        }
        
        res = _itypenew(&ITypeType,ntc,length,_common_subtype((IType *)args1->subtype,(IType *)args2->subtype));
    }
    else
    {
        if(args1->typecode == ntc)
        {
            Py_INCREF(args1);
            res = (PyObject *)args1;
        }
        else if(args2->typecode == ntc)
        {
            Py_INCREF(args2);
            res = (PyObject *)args2;
        }
        else
            res = _itypenew(&ITypeType,ntc,NULL,NULL);
        
    }
    return res; 

}

static PyObject * _getseqtype(PyObject * self, PyObject * args, it_code typecode,int depth)
{
    

    PyObject * seqtc;
    PyObject * res;
    PyObject * iter;
    it_code ntc;

    if(depth > 10)
    {
        PyErr_SetString(PyExc_TypeError,"Data too deeply nested.");
        return 0;
    }

    if(IS_TYPE(typecode,NUMPY))
    {
        PyObject * dtype;
        dtype = PyObject_GetAttrString(args,"dtype");
        if(dtype == NULL)
        {
            PyErr_SetString(PyExc_TypeError,"Numpy scalar has no dtype");
            return NULL;
        }
        seqtc = _dtype_to_typecode(dtype);
        if(seqtc == NULL)
            return NULL;
        ntc = PyInt_AsLong(seqtc);
        if(PyErr_Occurred())
            return NULL;

        if(IS_TYPE(ntc,~(OBJECT | UNKNOWN | SEQUENCE)))
        {
            res = _itypenew(&ITypeType,ntc,NULL,NULL); 
            return res;
        }
    }
   
    iter = PyObject_GetIter(args);
    if(iter == NULL || PyErr_Occurred() || !PyIter_Check(iter))
        return NULL;
    PyObject *item;
    PyObject *typelist = PyList_New(0);
    PyObject * elemlength = NULL;
    PyObject *subtype = NULL;
    
    int curlength = 0;
    int i;

    if(typelist == NULL)
        return NULL;

    while((item = PyIter_Next(iter)))
    {
        for(i = 0; i < curlength; i++)
            if(PyList_GET_ITEM(typelist,i) == (PyObject *)item->ob_type)
                break;
        if(i == curlength) //not found
        {
            res = register_pytype(item);
            if(res == NULL)
                return NULL;
            Py_DECREF(res);
            
            if(PyList_Append(typelist,(PyObject *)item->ob_type) == -1)
                return NULL;
            curlength += 1;
        }
        Py_DECREF(item);
    }
    Py_DECREF(iter);
    ntc = ~0;
    if(curlength > 0)
    {
        it_code tmp;
        for(i = 0; i < curlength; i++)
        {
            item = PyList_GET_ITEM(typelist,i);
            res = PyDict_GetItem(convdict,item);
            if(res == NULL)
                return NULL;
            tmp = PyInt_AsLong(res);
            ntc &= get_compatible_types(tmp);
        }
        if(PyErr_Occurred())
            return NULL;
    }
    else
        ntc = UNKNOWN;
   

    Py_DECREF(typelist);
   

    ntc = choose_type(ntc);
    if(IS_TYPE(ntc,SEQUENCE))
    {
        long maxlength = 0;
        long ilength = 0;
        PyObject *isubtype= NULL,*tmp;
        
        iter = PyObject_GetIter(args);
        if(iter == NULL || PyErr_Occurred() || !PyIter_Check(iter))
            return NULL;
        subtype = NULL;
        while((item = PyIter_Next(iter)))
        {
            ilength = PySequence_Length(item);
            if(ilength > maxlength) maxlength = ilength;

            if(IS_TYPE(ntc,RSEQUENCE))
            {
                isubtype = _getseqtype(self,item,ntc,depth + 1);
                if(isubtype == NULL)
                    return NULL;
                tmp = _common_subtype((IType *) subtype,(IType *)isubtype);
                Py_XDECREF(subtype);
                Py_XDECREF(isubtype);
                subtype = tmp;
            }
            Py_DECREF(item);
        }
        Py_DECREF(iter);
        elemlength = PyInt_FromLong(maxlength);
    }
   
    
    res = _itypenew(&ITypeType,ntc,elemlength,subtype);
    Py_XDECREF(subtype);
    return res;
}



static PyObject * _dtype_to_typecode(PyObject *dtype)
{
    PyObject *typechar, *typename, *typecode = NULL;
    char * buftc;
    /* Use dtype.char to filter out strings, unicode, and bytes arrays */
    
    typechar = PyObject_GetAttrString(dtype,"char");
    if(typechar == NULL || !PyString_Check(typechar) || PyString_GET_SIZE(typechar) == 0)
    {
        PyErr_SetString(PyExc_TypeError,"Numpy scalar dtype has no char.");
        Py_XDECREF(typechar);
        return NULL;
    }
    buftc = PyString_AsString(typechar);
    if(buftc == NULL)
    {
        PyErr_SetString(PyExc_TypeError,"Could not obtain typechar string buffer");
        Py_DECREF(typechar);
        return NULL;
    }
    switch(buftc[0])
    {
        case 'S':
            typecode = Py_typecodes[ISTR];
            break;
        case 'U':
            typecode = Py_typecodes[IUNICODE];
            break;
        case 'V':
            typecode = Py_typecodes[IBYTES];
            break;
    }       
    Py_DECREF(typechar);
    
    if(typecode != NULL)
    {
        Py_INCREF(typecode);
        return typecode;
    }

        
    /* Something else, look at typename */       
    typename = PyObject_GetAttrString(dtype,"name");
    if(typename == NULL || !PyString_Check(typename))
    {
        PyErr_SetString(PyExc_TypeError,"Numpy scalar dtype has no name.");
        Py_XDECREF(typename);
        return NULL;
    }
    typecode = PyDict_GetItem(typedict,typename);
    if(typecode == NULL)
    {
        PyErr_SetString(PyExc_TypeError,"Unknown numpy dtype encountered.");
        Py_DECREF(typename);
        return NULL;
    }
    Py_DECREF(typename);
    Py_INCREF(typecode);
    return typecode;

}


/* Converts types to a typecode, caches in dictionary */
static PyObject * register_pytype(PyObject *obj)
{
    int res;
    PyObject * typecode = NULL;
    typecode = PyDict_GetItem(convdict,(PyObject *)obj->ob_type);
    if(typecode != NULL)
    {
        Py_INCREF(typecode);
        return typecode;
    }

    if(PyArray_IsAnyScalar(obj) && !PyArray_Check(obj))
    {
        PyObject * dtype;
        
        if(PyArray_CheckScalar(obj))
        {
            dtype = PyObject_GetAttrString(obj,"dtype");
            if(dtype == NULL)
            {
                PyErr_SetString(PyExc_TypeError,"Numpy scalar has no dtype");
                return NULL;
            }
        }
        else if(PyArray_IsPythonScalar(obj))
        {
            if(PyArray_DescrConverter((PyObject *) obj->ob_type,(PyArray_Descr **)&dtype) == 0)
            {
                PyErr_SetString(PyExc_TypeError,"Could not determine numpy dtype for pythons scalar.");
                return NULL;
            }
        }
        typecode = _dtype_to_typecode(dtype);
        Py_DECREF(dtype);
        if(typecode == NULL)
            return NULL;
        Py_DECREF(typecode);  //warning: ugly, but further on it is increfed again, and _dtype_to_typecode borrows the ref from somewhere else
    }
    else if(PyArray_Check(obj))
        typecode = Py_typecodes[INUMPY];
    else if(PyTuple_Check(obj))
        typecode = Py_typecodes[ITUPLE];
    else if(PyList_Check(obj))
        typecode = Py_typecodes[ILIST];
    else if(PyString_Check(obj))
        typecode = Py_typecodes[ISTR];
    else if(PyUnicode_Check(obj))
        typecode = Py_typecodes[IUNICODE];
    else if(PyObject_IsInstance((PyObject *) obj->ob_type,(PyObject *)BaseContainer_Type))
        typecode = Py_typecodes[ICONTAINER];
    else
        typecode = Py_typecodes[IOBJECT];
   
    res = PyDict_SetItem(convdict,(PyObject *) obj->ob_type,typecode); 
    if(res == -1) return NULL;
    Py_INCREF(typecode);
    return typecode;
}


#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
inititypes(void) 
{
    PyObject* m;
    int i;
    m = Py_InitModule3("itypes", module_methods,
                       "Type handling module for ibidas");

    import_array();
    import_base_container();
    if (PyType_Ready(&ITypeType) < 0)
        return;

    if (m == NULL)
      return;

    Py_INCREF(&ITypeType);
    PyModule_AddObject(m, "IType", (PyObject *)&ITypeType);
    
    
    typedict =  PyDict_New();
    if(typedict == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,"Type dictionary could not be created.");
        return;
    }
    for(i = 0; i < NRTYPE; i++)
    {
        Py_typecodes[i] = PyInt_FromLong(1 << i);
        Py_defval[i] = get_defval(1 << i);
        if(Py_typecodes[i] == NULL)
            goto error_tc;

        if(PyDict_SetItemString(typedict,type_names[i],Py_typecodes[i]) < 0)
            goto error_tc;
        
    }
    if(PyDict_SetItemString(typedict,"SEQUENCE",PyInt_FromLong(SEQUENCE)) < 0)
        goto error_tc;
    if(PyDict_SetItemString(typedict,"RSEQUENCE",PyInt_FromLong(RSEQUENCE)) < 0)
        goto error_tc;
    if(PyDict_SetItemString(typedict,"UINT",PyInt_FromLong(UINT)) < 0)
        goto error_tc;
    if(PyDict_SetItemString(typedict,"INT",PyInt_FromLong(INT)) < 0)
        goto error_tc;
    if(PyDict_SetItemString(typedict,"INTEGER",PyInt_FromLong(INTEGER)) < 0)
        goto error_tc;
    if(PyDict_SetItemString(typedict,"FLOAT",PyInt_FromLong(FLOAT)) < 0)
        goto error_tc;
    if(PyDict_SetItemString(typedict,"COMPLEX",PyInt_FromLong(COMPLEX)) < 0)
        goto error_tc;
    if(PyDict_SetItemString(typedict,"NUMBER",PyInt_FromLong(NUMBER)) < 0)
        goto error_tc;


    Py_INCREF(typedict);
    PyModule_AddObject(m, "types", typedict);
    convdict = PyDict_New();
    if(convdict == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,"Conversion dictionary could not be created");
        return;
    }
    return;

error_tc:
    PyErr_SetString(PyExc_RuntimeError,"Initialization of typecodes failed.");
    return;
}



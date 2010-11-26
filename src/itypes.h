#include <Python.h>
#include "structmember.h"
#include <string.h>
#include "numpy/arrayobject.h"


#define MAX(x,y)       (((x) > (y)) ? x : y)

#define BIT(x)  (0x00000001 << (x))

#define IINT8        0
#define IINT16       1
#define IINT32       2
#define IINT64       3

#define IUINT8       4
#define IUINT16      5
#define IUINT32      6
#define IUINT64      7

#define IFLOAT32     8
#define IFLOAT64     9
#define IFLOAT128    10
#define ICOMPLEX64   11

#define ICOMPLEX128  12
#define ICOMPLEX256  13
#define IBOOL        14
#define ITUPLE       15

#define ILIST        16
#define IARRAY       17
#define INUMPY       18
#define ICONTAINER   19

#define ISTR         20
#define IUNICODE     21
#define IBYTES       22
#define IOBJECT      23

#define IUNKNOWN     24


#define INT8        BIT(IINT8)
#define INT16       BIT(IINT16)
#define INT32       BIT(IINT32)
#define INT64       BIT(IINT64)

#define UINT8       BIT(IUINT8)
#define UINT16      BIT(IUINT16)
#define UINT32      BIT(IUINT32)
#define UINT64      BIT(IUINT64)

#define FLOAT32     BIT(IFLOAT32)
#define FLOAT64     BIT(IFLOAT64)
#define FLOAT128    BIT(IFLOAT128)
#define COMPLEX64   BIT(ICOMPLEX64)

#define COMPLEX128  BIT(ICOMPLEX128)
#define COMPLEX256  BIT(ICOMPLEX256)
#define BOOL        BIT(IBOOL)
#define TUPLE       BIT(ITUPLE)

#define LIST        BIT(ILIST)
#define ARRAY       BIT(IARRAY)
#define NUMPY       BIT(INUMPY)
#define CONTAINER   BIT(ICONTAINER)

#define STR         BIT(ISTR)
#define UNICODE     BIT(IUNICODE)
#define BYTES       BIT(IBYTES)
#define OBJECT      BIT(IOBJECT)

#define UNKNOWN     BIT(IUNKNOWN)

#define SEQUENCE    (TUPLE | LIST | ARRAY | NUMPY | CONTAINER | STR | UNICODE | BYTES)
#define RSEQUENCE   (TUPLE | LIST | ARRAY | NUMPY | CONTAINER)
#define UINT        (UINT8 | UINT16 | UINT32 | UINT64)
#define INT         (INT8 | INT16 | INT32 | INT64)
#define INTEGER     (INT | UINT)
#define FLOAT       (FLOAT32 | FLOAT64 | FLOAT128)
#define COMPLEX     (COMPLEX64 | COMPLEX128 | COMPLEX256)
#define NUMBER      (BOOL | INTEGER | FLOAT | COMPLEX)

#define NOTYPE      (BIT(25) | BIT(26) | BIT(27) | BIT(28) | BIT(28) | BIT(29) | BIT(30) | BIT(31))
#define NRTYPE      25

#define IS_TYPE(x,y) ((((x) & (y)) && !((x) & ~(y))) ? 1 : 0)
#define VALID_TYPE(x) (((bitcount((x)) == 1) && !(x & NOTYPE)) ? 1 : 0)


#define L_UNKNOWN -1

static char * type_names[NRTYPE] = {"int8","int16","int32","int64","uint8","uint16",
                     "uint32","uint64","float32","float64","float128",
                     "complex64","complex128","complex256","bool",
                     "tuple","list","array","numpy","container",
                     "str","unicode","bytes","object","unknown"};
static char * numpy_names[NRTYPE] = {"int8","int16","int32","int64","uint8","uint16",
                     "uint32","uint64","float32","float64","float128",
                     "complex64","complex128","complex256","bool",
                     "object","object","object","object","object",
                     "S","U","V","object","object"};
static char type_priority[NRTYPE] = {102,92,82,72,  //ints
                                     103,93,83,73,  //uints
                                     81,71,61,     //floats
                                     70,60,50,     //complex
                                     114,           //bool
                                     40,39,41,42,38,//seqs
                                     20,15,11,10,0}; //str,unicode,bytes,object,unknown
PyObject * Py_typecodes[NRTYPE];
PyObject * Py_defval[NRTYPE];
PyObject * convdict = NULL;
PyObject * typedict = NULL;

typedef unsigned long it_code;

static it_code choose_type(it_code typecode)
{
    char curprio = -1;
    char curidx = -1;
    int i;
    for(i = 0; i < NRTYPE; i++)
        if(typecode & BIT(i))
            if(type_priority[i] > curprio)
            {
                curidx = i;
                curprio = type_priority[i];
            }
    if(curidx == -1)
        return 0;
    else
        return BIT(curidx);
}


static it_code get_convertable_types(it_code typecode)
{   
    it_code res = 0;
    switch(typecode)
    {
        case INT8:      res = OBJECT | INT | FLOAT | COMPLEX; break;
        case INT16:     res = ((OBJECT | INT) & ~INT8) | FLOAT | COMPLEX; break;
        case INT32:     res = OBJECT | INT64 | FLOAT64 | FLOAT128 | COMPLEX128 | COMPLEX256; break;
        case INT64:     res = OBJECT | FLOAT64 | FLOAT128 | COMPLEX128 | COMPLEX256; break;
        case UINT8:     res = ((OBJECT | INTEGER) & ~INT8) | FLOAT | COMPLEX; break;
        case UINT16:    res = ((OBJECT | INTEGER) & ~(UINT8 | INT8 | INT16)) | FLOAT | COMPLEX; break;
        case UINT32:    res = OBJECT | UINT64 | INT64 | FLOAT64 | FLOAT128 | COMPLEX128 | COMPLEX256; break;
        case UINT64:    res = OBJECT | FLOAT64 | FLOAT128 | COMPLEX128 | COMPLEX256; break;
        case FLOAT32:   res = OBJECT | FLOAT64 | FLOAT128 | COMPLEX; break;
        case FLOAT64:   res = OBJECT | FLOAT128 | COMPLEX128 | COMPLEX256; break;
        case FLOAT128:  res = OBJECT | COMPLEX256; break;
        case COMPLEX64: res = OBJECT | COMPLEX128 | COMPLEX256; break;
        case COMPLEX128:res = OBJECT | COMPLEX256; break;
        case COMPLEX256:res = OBJECT; break;
        case BOOL:      res = OBJECT | INTEGER | FLOAT | COMPLEX; break;
        case TUPLE:     
        case LIST:      
        case ARRAY:     
        case NUMPY:     
        case CONTAINER: 
        case STR:       
        case UNICODE:   
        case BYTES:     
        case OBJECT:    
        case UNKNOWN:   res = OBJECT; break;
    }
    res |= typecode;
    return res;
}

static it_code get_compatible_types(it_code typecode)
{   
    it_code res = 0;
    switch(typecode)
    {
        case INT8:      res = OBJECT | INT; break;
        case INT16:     res = (OBJECT | INT) & ~INT8; break;
        case INT32:     res = OBJECT | INT64; break;
        case INT64:     res = OBJECT; break;
        case UINT8:     res = (OBJECT | INTEGER) & ~INT8; break;
        case UINT16:    res = (OBJECT | INTEGER) & ~(UINT8 | INT8 | INT16); break;
        case UINT32:    res = OBJECT | UINT64 | INT64; break;
        case UINT64:    res = OBJECT; break;
        case FLOAT32:   res = OBJECT | FLOAT64 | FLOAT128; break;
        case FLOAT64:   res = OBJECT | FLOAT128; break;
        case FLOAT128:  res = OBJECT; break;
        case COMPLEX64: res = OBJECT | COMPLEX128 | COMPLEX256; break;
        case COMPLEX128:res = OBJECT | COMPLEX256; break;
        case COMPLEX256:res = OBJECT; break;
        case BOOL:      res = OBJECT | INTEGER; break;
        case TUPLE:     
        case LIST:      
        case ARRAY:     
        case NUMPY:     
        case CONTAINER: 
        case STR:       
        case UNICODE:   
        case BYTES:     
        case OBJECT:    
        case UNKNOWN:   res = OBJECT; break;
    }
    res = res | typecode;
    return res;
}


int bitcount (it_code typecode)
{
    int count=0 ;
    while (typecode)
    {
        count++ ;
        typecode &= (typecode - 1) ;     
    }
    return count ;
}

static unsigned char get_type_nr(it_code typecode)
{
    switch(typecode)
    {
        case INT8:      return IINT8;
        case INT16:     return IINT16;
        case INT32:     return IINT32;
        case INT64:     return IINT64;
        case UINT8:     return IUINT8;
        case UINT16:    return IUINT16;
        case UINT32:    return IUINT32;
        case UINT64:    return IUINT64;
        case FLOAT32:   return IFLOAT32;
        case FLOAT64:   return IFLOAT64;
        case FLOAT128:  return IFLOAT128;
        case COMPLEX64: return ICOMPLEX64;
        case COMPLEX128:return ICOMPLEX128;
        case COMPLEX256:return ICOMPLEX256;
        case BOOL:      return IBOOL;
        case TUPLE:     return ITUPLE;
        case LIST:      return ILIST;
        case ARRAY:     return IARRAY;
        case NUMPY:     return INUMPY;
        case CONTAINER: return ICONTAINER;
        case STR:       return ISTR;
        case UNICODE:   return IUNICODE;
        case BYTES:     return IBYTES;
        case OBJECT:    return IOBJECT;
        case UNKNOWN:   return IUNKNOWN;
    }
    return IUNKNOWN;
}

static PyObject * get_defval(it_code typecode)
{
    switch(typecode)
    {
        case INT8:      return PyInt_FromLong(0);
        case INT16:     return PyInt_FromLong(0);
        case INT32:     return PyInt_FromLong(0);
        case INT64:     return PyInt_FromLong(0);
        case UINT8:     return PyInt_FromLong(0);
        case UINT16:    return PyInt_FromLong(0);
        case UINT32:    return PyInt_FromLong(0);
        case UINT64:    return PyInt_FromLong(0);
        case FLOAT32:   return PyFloat_FromDouble(0.0);
        case FLOAT64:   return PyFloat_FromDouble(0.0);
        case FLOAT128:  return PyFloat_FromDouble(0.0);
        case COMPLEX64: return PyComplex_FromDoubles(0.0,0.0);
        case COMPLEX128:return PyComplex_FromDoubles(0.0,0.0);
        case COMPLEX256:return PyComplex_FromDoubles(0.0,0.0);
        case BOOL:      return PyBool_FromLong(0);
        case TUPLE:     Py_RETURN_NONE;
        case LIST:      Py_RETURN_NONE;
        case ARRAY:     Py_RETURN_NONE;
        case NUMPY:     Py_RETURN_NONE;
        case CONTAINER: Py_RETURN_NONE;
        case STR:       return PyString_FromString("");
        case UNICODE:   return PyUnicode_DecodeASCII("",0,NULL);
        case BYTES:     Py_RETURN_NONE;
        case OBJECT:    Py_RETURN_NONE;
        case UNKNOWN:   Py_RETURN_NONE;
        default:        Py_RETURN_NONE;
    }
}

#define STATE_START 0
#define STATE_INSUBTYPE 1
#define STATE_ENDSUBTYPE 2
#define STATE_INBLOCK 3
#define STATE_ENDBLOCK 4
#define STATE_SUBTYPEINBLOCK 5

static PyObject * _getseqtype(PyObject * self, PyObject * args, it_code typecode,int depth);
static PyObject * _dtype_to_typecode(PyObject *dtype);
static PyObject * IType_from_numpy(PyObject *self, PyObject *obj);
static PyObject * register_pytype(PyObject *obj);
static PyObject * _callback_wrapper(PyObject *self, PyObject *args, PyCFunction callback);


#include <Python.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include "SFMT.h"
#include <xmmintrin.h>
#define L_UNKNOWN -1
#define FHIERARCH 1
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )
static inline char popcount_wegner(uint64_t v) {
    char cnt=0;
    while (v) {
        cnt++;
        v &= v-1;  /*  clear the least significant bit set */
    }
    return cnt;
}

PyObject * random_module;

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

static PyArray_Descr * get_descriptor(const char * descr)
{
    PyObject * tmp;
    PyArray_Descr * tdescr;

    tmp = PyString_FromString(descr);
    if(tmp == NULL)
        return NULL;

    PyArray_DescrConverter(tmp,&tdescr);
    Py_DECREF(tmp);
    return tdescr;
}


static PyObject * aligned_np_vector(PyArray_Descr * tdescr, npy_intp size, size_t boundary)
{
    PyObject * tmp, * result;
    void * result_ptr;
    size_t offset;
    npy_intp  buffersize = size + boundary;

    Py_INCREF(tdescr); 
    tmp = PyArray_NewFromDescr(&PyArray_Type, tdescr, 1, &buffersize, NULL, NULL, 0, NULL);
    if(tmp == NULL)
        return NULL;

    result_ptr = PyArray_DATA(tmp);
    offset = boundary - ((size_t)result_ptr % boundary);
    result = PySequence_GetSlice(tmp, offset, offset + size);
    Py_DECREF(tmp);

    return result;
}

#define rbits(name,n64,n64_mask,n64_dtype) \
static PyObject * random_bits##name(PyObject *self, PyObject *count_seq)  \
{ \
    PyArray_Descr * tdescr; \
    PyObject * result; \
    uint8_t * array, *array_ptr, *max_ptr; \
    size_t random_array_size, count_seq_size; \
    int i; \
 \
    uint64_t new_value[n64], nnew_value; \
    uint64_t *result_ptr; \
    uint64_t count, *count_seq_ptr, *count_seq_end; \
    uint8_t cur_pos, negative=0; \
 \
    PyObject * getrandbits = PyObject_GetAttrString(random_module, "getrandbits"); \
    PyObject * res = PyObject_CallFunction(getrandbits, "i" ,32); \
    Py_DECREF(getrandbits); \
 \
    if(!PyLong_Check(res)) \
        return NULL; \
    unsigned long seed = PyLong_AsUnsignedLong(res);  \
    Py_DECREF(res);  \
    init_gen_rand((unsigned int) (seed & 0xffffffff)); \
   \
    if(!PyArray_Check(count_seq) || !PyArray_ISINTEGER(count_seq) || \
       !PyArray_ISUNSIGNED(count_seq) || ((PyArrayObject *)count_seq)->nd != 1 || \
       !PyArray_ISCONTIGUOUS(count_seq) || ((PyArrayObject *)count_seq)->descr->elsize != 8) \
    { \
        PyErr_SetString(PyExc_TypeError, "Input should be a contiguous 1-dimensional numpy array of uint64, containing bit counts."); \
        return NULL;  \
    }  \
    count_seq_size = PyArray_SIZE(count_seq); \
    \
    tdescr = get_descriptor(n64_dtype); \
    if(tdescr == NULL)    \
        return NULL; \
    result = aligned_np_vector(tdescr, count_seq_size, 16);   \
    if(result == NULL)   \
        return NULL; \
    Py_DECREF(tdescr); \
    \
    random_array_size = 4096 + 256 + n64 * 64; \
    \
    if (posix_memalign((void **)&array, 16, sizeof(double) * random_array_size) != 0)  \
    { \
        PyErr_SetString(PyExc_TypeError, "Cannot allocate aligned memory"); \
        Py_DECREF(result); \
        return NULL; \
    }    \
    if (array == NULL)  \
    { \
        PyErr_SetString(PyExc_TypeError, "Cannot allocate aligned memory"); \
        Py_DECREF(result); \
        return NULL; \
    } \
    \
    fill_array64((uint64_t *)array, random_array_size); \
    array_ptr = array; \
    max_ptr = array + random_array_size * sizeof(double) - (256 + n64 * 64); \
    \
    result_ptr = PyArray_DATA(result); \
    count_seq_ptr = PyArray_DATA(count_seq); \
    count_seq_end = count_seq_ptr + count_seq_size; \
    while(count_seq_ptr < count_seq_end) \
    { \
        count = *count_seq_ptr; \
        if(count > n64 * 32) \
        { \
            if(count > n64 * 64) \
            { \
                PyErr_SetString(PyExc_TypeError, "Values in input array should be <= " #name); \
                Py_DECREF(result); \
                free(array); \
                return NULL; \
            } \
            count = n64 * 64 - count; \
            negative = 1; \
        } \
       \
        if(array_ptr > max_ptr) \
        { \
            fill_array64((uint64_t *)array, (((array_ptr - array)/sizeof(double)) + 2) & ~0x1); \
            array_ptr = array; \
        }  \
        for(i = 0; i < n64; i++) \
            new_value[i] = 0; \
        \
        while(count) \
        { \
            cur_pos = *(array_ptr++); \
            i = cur_pos & n64_mask; \
            nnew_value = new_value[i] |(((uint64_t) 1) << (cur_pos >> 2)); \
            count -= (nnew_value != new_value[i]); \
            new_value[i] = nnew_value; \
        } \
        \
        if(negative) \
        { \
            for(i = 0; i < n64; i++) \
            { \
                *result_ptr = ~new_value[i]; \
                result_ptr++; \
            } \
            negative = 0; \
        } \
        else \
        { \
            for(i = 0; i < n64; i++) \
            { \
              *result_ptr = new_value[i]; \
              result_ptr++; \
            } \
        } \
        count_seq_ptr++; \
    }; \
    free(array); \
    return result;  \
} \

rbits(64,1,0x00,"u8") 
rbits(128,2,0x01,"u8, u8") 
rbits(256,4,0x03,"u8, u8, u8, u8") 

#define rbits_input(name,n64, n64_mask, n64_dtype) \
static PyObject * random_bits_input##name(PyObject *self, PyObject *args)  \
{ \
    PyArray_Descr * tdescr; \
    PyObject * result; \
    uint8_t * array, *array_ptr, *max_ptr, *array_ptr_max; \
    size_t random_array_size, count_seq_size, count_seq_done_size, input_size, groupidx_size; \
    npy_intp *grouporder_size;\
    int i, flags=0; \
 \
    uint64_t nnew_value, totcount; \
    uint64_t *result_ptr, *grouporder_cur, *grouporder_ptr; \
    uint64_t count, count_done, *count_ptr, *count_seq_end, *input_ptr, *count_done_ptr; \
    int64_t group, * groupidx_ptr; \
    uint8_t pos; \
\
    uint8_t cur_pos, negative=0; \
    \
    PyObject *count_seq_done, *count_seq, *input, *groupidx, *grouporder;\
    \
    if(!PyArg_ParseTuple(args,"OOOOO|i",&count_seq_done, &count_seq, &input, &groupidx, &grouporder, &flags))\
        return NULL;\
 \
    PyObject * getrandbits = PyObject_GetAttrString(random_module, "getrandbits"); \
    PyObject * res = PyObject_CallFunction(getrandbits, "i" ,32); \
    Py_DECREF(getrandbits); \
 \
    if(!PyLong_Check(res)) \
        return NULL; \
    unsigned long seed = PyLong_AsUnsignedLong(res);  \
    Py_DECREF(res);  \
    init_gen_rand((unsigned int) (seed & 0xffffffff)); \
 \
\
    if(!PyArray_Check(input) || ((PyArrayObject *)input)->nd != 1 || \
       !PyArray_ISCONTIGUOUS(input) || ((PyArrayObject *)input)->descr->elsize != (n64 * 8)) \
    { \
        PyErr_SetString(PyExc_TypeError, "input should be a contiguous 1-dimensional numpy array of uint64s, containing bernoulli bits."); \
        return NULL;  \
    }  \
    input_size = PyArray_SIZE(input); \
    \
    if(!PyArray_Check(count_seq) || !PyArray_ISINTEGER(count_seq) || \
       !PyArray_ISUNSIGNED(count_seq) || ((PyArrayObject *)count_seq)->nd != 1 || \
       !PyArray_ISCONTIGUOUS(count_seq) || ((PyArrayObject *)count_seq)->descr->elsize != 8) \
    { \
        PyErr_SetString(PyExc_TypeError, "count_seq should be a contiguous 1-dimensional numpy array of uint64, containing bit counts."); \
        return NULL;  \
    }  \
    count_seq_size = PyArray_SIZE(count_seq); \
\
    if(!PyArray_Check(count_seq_done) || !PyArray_ISINTEGER(count_seq_done) || \
       !PyArray_ISUNSIGNED(count_seq_done) || ((PyArrayObject *)count_seq_done)->nd != 1 || \
       !PyArray_ISCONTIGUOUS(count_seq_done) || ((PyArrayObject *)count_seq_done)->descr->elsize != 8) \
    { \
        PyErr_SetString(PyExc_TypeError, "count_seq_done should be a contiguous 1-dimensional numpy array of uint64, containing bit counts."); \
        return NULL;  \
    }  \
    count_seq_done_size = PyArray_SIZE(count_seq_done); \
\
    if(!PyArray_Check(groupidx) || !PyArray_ISINTEGER(groupidx) || \
       !PyArray_ISSIGNED(groupidx) || ((PyArrayObject *)groupidx)->nd != 1 || \
       !PyArray_ISCONTIGUOUS(groupidx) || ((PyArrayObject *)groupidx)->descr->elsize != 8) \
    { \
        PyErr_SetString(PyExc_TypeError, "groupidx should be a contiguous 1-dimensional numpy array of int64, containing group indexes."); \
        return NULL;  \
    }  \
    groupidx_size = PyArray_SIZE(groupidx); \
    \
    if(!PyArray_Check(grouporder) || !PyArray_ISINTEGER(grouporder) || \
       !PyArray_ISUNSIGNED(grouporder) || ((PyArrayObject *)grouporder)->nd != 2 || \
       !PyArray_ISCONTIGUOUS(grouporder) || ((PyArrayObject *)grouporder)->descr->elsize != 8) \
    { \
        PyErr_SetString(PyExc_TypeError, "grouporder should be a contiguous 2-dimensional numpy array of uint64, containing group field orders."); \
        return NULL;  \
    }  \
    grouporder_size = PyArray_DIMS(grouporder); \
\
    if(!(count_seq_size == count_seq_done_size && count_seq_size == groupidx_size && count_seq_size == input_size))\
    {\
        PyErr_SetString(PyExc_TypeError, "Count_seq, group_idx, input_seq and count_seq_done should have the same length."); \
        return NULL;  \
    }\
    \
    if(!(grouporder_size[1] == n64 * 64))\
    {\
        PyErr_SetString(PyExc_TypeError, "grouporder.dim2 should be equal to number of bits."); \
        return NULL;  \
    }\
    \
    tdescr = get_descriptor(n64_dtype); \
    if(tdescr == NULL)    \
        return NULL; \
    result = aligned_np_vector(tdescr, count_seq_size, 16);   \
    if(result == NULL)   \
        return NULL; \
    Py_DECREF(tdescr); \
    \
    random_array_size = 4096 + 1024 + n64 * 64; \
    \
    if (posix_memalign((void **)&array, 16, sizeof(double) * random_array_size) != 0)  \
    { \
        PyErr_SetString(PyExc_TypeError, "Cannot allocate aligned memory"); \
        Py_DECREF(result); \
        return NULL; \
    }    \
    if (array == NULL)  \
    { \
        PyErr_SetString(PyExc_TypeError, "Cannot allocate aligned memory"); \
        Py_DECREF(result); \
        return NULL; \
    } \
    \
    fill_array64((uint64_t *)array, random_array_size); \
    array_ptr = array; \
    max_ptr = array + random_array_size * sizeof(double) - (1024 + n64 * 64); \
    \
    result_ptr = PyArray_DATA(result); \
    count_ptr = PyArray_DATA(count_seq);\
    count_done_ptr = PyArray_DATA(count_seq_done);\
    groupidx_ptr = PyArray_DATA(groupidx);\
    grouporder_ptr = PyArray_DATA(grouporder);\
    input_ptr = PyArray_DATA(input);\
\
    count_seq_end = count_ptr + count_seq_size; \
    while(count_ptr < count_seq_end) \
    { \
        count = *(count_ptr++); \
        count_done = *(count_done_ptr++);\
        group = *(groupidx_ptr++);\
        if(count == count_done)\
        {\
            memcpy(result_ptr,input_ptr,sizeof(uint64_t) * n64);\
            result_ptr += n64;\
            input_ptr += n64;\
            continue;\
        }\
\
        if(count > n64 * 64) \
        { \
            PyErr_SetString(PyExc_TypeError, "Values in input array should be <= " #name); \
            Py_DECREF(result); \
            free(array); \
            return NULL; \
        } \
\
        if((count_done + (n64 * 64 - count)) < count)\
        {\
            count = (n64 * 64 ) - count;\
            negative = 1;\
        }\
        else\
        {\
            count = count - count_done;\
            if(count <= 0) \
            { \
                    PyErr_SetString(PyExc_TypeError, "Count values should be >= count_done values" ); \
                    Py_DECREF(result); \
                    free(array); \
                    return NULL; \
            } \
        }\
        memcpy(result_ptr, input_ptr, sizeof(uint64_t) * n64);\
        \
        if((count + count_done) < n64 * 53)\
        {\
            if(array_ptr > max_ptr) \
            { \
                if(array_ptr > (array + random_array_size * sizeof(double))) \
                {\
                    printf("Array bounds violated by %lu!\n", array_ptr - array - random_array_size * sizeof(double)); \
                    return NULL; \
                } \
                fill_array64((uint64_t *)array, (((array_ptr - array)/sizeof(double)) + 2) & ~0x1); \
                array_ptr = array; \
            }\
            while(count) \
            { \
                cur_pos = *(array_ptr++); \
                i = cur_pos & n64_mask; \
                nnew_value = result_ptr[i] |(((uint64_t) 1) << (cur_pos >> 2)); \
                count -= (nnew_value != result_ptr[i]); \
                result_ptr[i] = nnew_value; \
            }\
            if(negative)\
            {\
                for(i = 0; i < n64; i++) \
                { \
                    result_ptr[i] = input_ptr[i] | ~result_ptr[i]; \
                }\
                negative = 0;\
            }\
        }\
        else\
        {\
            \
            if(group >= grouporder_size[0])\
            {\
                PyErr_SetString(PyExc_TypeError, "Group index larger then available rows in grouporder"); \
                Py_DECREF(result); \
                free(array); \
                return NULL; \
            }\
\
            grouporder_cur = &(grouporder_ptr[(n64 * 64) * group]);\
\
            if(flags & FHIERARCH)\
            {\
                count_done = n64 * 64;\
                \
                totcount = 0;\
                pos = 0; \
                if(array_ptr > max_ptr)\
                {\
                    fill_array64((uint64_t *)array, (((array_ptr - array)/sizeof(double)) + 2) & ~0x1); \
                    array_ptr = array; \
                }\
                while(count) \
                { \
                    pos = (pos + (*(array_ptr++))) % count_done; \
                    count_done--;\
                    cur_pos = grouporder_cur[pos]; \
                    grouporder_cur[pos] = grouporder_cur[count_done];\
                    grouporder_cur[count_done] = cur_pos;\
                    i = cur_pos & n64_mask; \
                    nnew_value = result_ptr[i] |(((uint64_t) 1) << (cur_pos >> 2)); \
                    count -= (nnew_value != result_ptr[i]); \
                    result_ptr[i] = nnew_value; \
                }\
            }\
            else\
            {\
                count_done = n64 * 64 - count_done;\
                \
                totcount = 0;\
                pos = 0; \
                while(count)\
                {\
                    array_ptr_max = array_ptr + count;\
                    if(array_ptr_max > max_ptr) \
                    { \
                        fill_array64((uint64_t *)array, (((array_ptr - array)/sizeof(double)) + 2) & ~0x1); \
                        array_ptr = array; \
                        array_ptr_max = array_ptr + count;\
                        \
                        if(totcount > n64 * 64 * 2)\
                        {\
                            PyErr_SetString(PyExc_TypeError, "Cannot find enough empty positions to fill bernoulli bits. Is grouporder array correct?"); \
                            Py_DECREF(result); \
                            free(array); \
                            return NULL; \
                        }\
                    }\
                    totcount += count;\
                    while(array_ptr < array_ptr_max) \
                    { \
                        pos = (pos + (*(array_ptr++))) % count_done; \
                        cur_pos = grouporder_cur[pos]; \
                        i = cur_pos & n64_mask; \
                        nnew_value = result_ptr[i] |(((uint64_t) 1) << (cur_pos >> 2)); \
                        count -= (nnew_value != result_ptr[i]); \
                        result_ptr[i] = nnew_value; \
                    }\
                }\
            }\
            if(negative)\
            {\
                for(i = 0; i < n64; i++) \
                { \
                    result_ptr[i] = input_ptr[i] | ~result_ptr[i]; \
                }\
                negative = 0;\
            }\
        }\
        result_ptr += n64;\
        input_ptr += n64;\
    } \
    free(array); \
    return result;  \
} \

rbits_input(64,1,0x00,"u8") 
rbits_input(128,2,0x01,"u8, u8") 
rbits_input(256,4,0x03,"u8, u8, u8, u8") 

#define n64 1
#define n64_dtype "u8"
#define n64_mask 0x00
#define name "64"
static PyObject * random_bits_corr(PyObject *self, PyObject * args)
{
    int i,j,k,flags;
    PyObject * score_seq, * prob_seq, *threshold_seq=NULL;
    PyObject * count_seq=NULL, *count_done_seq=NULL, *result=NULL;
    size_t score_seq_size, parent_seq_size, prob_seq_size, random_array_size, groupidx_seq_size;
    uint8_t *array, *array_ptr, *max_ptr, *max_ptr_cur;
    uint64_t *result_ptr, *groupresult_ptr, *count_ptr, *count_done_ptr, *groupidx_ptr, *grouporder_ptr;
    uint64_t * threshold_ptr;
    int64_t * parent_ptr;
    double *score_ptr, *prob_ptr;
    PyObject *groupresult_seq=NULL, *parent_seq=NULL, *groupidx_seq=NULL, *grouporder_seq=NULL;
    PyArray_Descr * tdescr;
    npy_intp grouporder_size[2];

    if(!PyArg_ParseTuple(args,"OOOOi",&prob_seq, &score_seq,&parent_seq, &groupidx_seq, &flags))
        return NULL;
   
    PyObject * getrandbits = PyObject_GetAttrString(random_module, "getrandbits"); 
    PyObject * res = PyObject_CallFunction(getrandbits, "i" ,32); 
    Py_DECREF(getrandbits); 
 
    if(!PyLong_Check(res)) 
        return NULL; 
    unsigned long seed = PyLong_AsUnsignedLong(res);  
    Py_DECREF(res);  
    init_gen_rand((unsigned int) (seed & 0xffffffff)); 
    
    if(!PyArray_Check(score_seq) || !PyArray_ISFLOAT(score_seq) || /*{{{*/
       ((PyArrayObject *)score_seq)->nd != 1 || 
       !PyArray_ISCONTIGUOUS(score_seq) || ((PyArrayObject *)score_seq)->descr->elsize != 8) 
    { 
        PyErr_SetString(PyExc_TypeError, "score_seq should be a contiguous 1-dimensional numpy array of doubles, containing correlation scores 0 <= 1 <= 1."); 
        return NULL;  
    }  
    score_seq_size = PyArray_SIZE(score_seq); 
    
    if(!PyArray_Check(prob_seq) || !PyArray_ISFLOAT(prob_seq) || 
       ((PyArrayObject *)prob_seq)->nd != 1 || 
       !PyArray_ISCONTIGUOUS(prob_seq) || ((PyArrayObject *)prob_seq)->descr->elsize != 8) 
    { 
        PyErr_SetString(PyExc_TypeError, "prob_seq should be a contiguous 1-dimensional numpy array of doubles, containing probilities 0 <= 1 <= 1."); 
        return NULL;  
    }  
    prob_seq_size = PyArray_SIZE(prob_seq); 
    
    if(!PyArray_Check(parent_seq) || !PyArray_ISINTEGER(parent_seq) || 
    !PyArray_ISSIGNED(parent_seq) || ((PyArrayObject *)parent_seq)->nd != 1 || 
    !PyArray_ISCONTIGUOUS(parent_seq) || ((PyArrayObject *)parent_seq)->descr->elsize != 8) 
    { 
        PyErr_SetString(PyExc_TypeError, "parent_seq should be a contiguous 1-dimensional numpy array of int64, containing parent indexes."); 
        return NULL;  
    }  
    parent_seq_size = PyArray_SIZE(parent_seq); 
    
    if(!PyArray_Check(groupidx_seq) || !PyArray_ISINTEGER(groupidx_seq) || 
    !PyArray_ISSIGNED(groupidx_seq) || ((PyArrayObject *)groupidx_seq)->nd != 1 || 
    !PyArray_ISCONTIGUOUS(groupidx_seq) || ((PyArrayObject *)groupidx_seq)->descr->elsize != 8) 
    { 
        PyErr_SetString(PyExc_TypeError, "groupidx_seq should be a contiguous 1-dimensional numpy array of int64, containing groupidx indexes."); 
        return NULL;  
    }  
    groupidx_seq_size = PyArray_SIZE(groupidx_seq); 

    if(!(groupidx_seq_size == prob_seq_size && parent_seq_size == score_seq_size))
    {
        PyErr_SetString(PyExc_TypeError, "parent_seq should be equal in length to score_seq, groupidx_seq to prob_seq."); 
        return NULL;  
    }/*}}}*/
    
    tdescr = get_descriptor("u8"); /*{{{*/
    if(tdescr == NULL)    
        return NULL; 
    
    count_seq = aligned_np_vector(tdescr, prob_seq_size, 16);   
    if(count_seq == NULL)   
        return NULL; 
    
    count_done_seq = aligned_np_vector(tdescr, prob_seq_size, 16);   
    if(count_done_seq == NULL)   
        return NULL; 
    
    grouporder_size[0] = parent_seq_size;
    grouporder_size[1] = n64 * 64;
    //eats tdescr
    grouporder_seq = PyArray_NewFromDescr(&PyArray_Type, tdescr, 2, grouporder_size, NULL, NULL, 0, NULL);
    tdescr = get_descriptor(n64_dtype); 
    groupresult_seq = PyArray_NewFromDescr(&PyArray_Type, tdescr, 2, grouporder_size, NULL, NULL, 0, NULL);
    
    
    tdescr = get_descriptor(n64_dtype); 
    result = aligned_np_vector(tdescr, prob_seq_size, 16);   
    if(result == NULL)   
        return NULL; 
    threshold_seq = aligned_np_vector(tdescr, n64 * 64, 16);   
    if(threshold_seq == NULL)   
        return NULL; 
    Py_DECREF(tdescr); 
    
/*}}}*/
    
    random_array_size = 4096 + 1024 + n64 * 64; 
    
    if (posix_memalign((void **)&array, 16, sizeof(double) * random_array_size) != 0)  
    { 
        PyErr_SetString(PyExc_TypeError, "Cannot allocate aligned memory"); 
        goto error;
    }    
    if (array == NULL)  
    { 
        PyErr_SetString(PyExc_TypeError, "Cannot allocate aligned memory"); 
        goto error;
    } 
    fill_array64((uint64_t *)array, random_array_size); 
    array_ptr = array; 
    max_ptr = array + random_array_size * sizeof(double) - (1024 + n64 * 64); 
    
    result_ptr = PyArray_DATA(result); 
    count_ptr = PyArray_DATA(count_seq); 
    count_done_ptr = PyArray_DATA(count_done_seq); 
    prob_ptr = PyArray_DATA(prob_seq); 
    score_ptr = PyArray_DATA(score_seq); 
    parent_ptr = PyArray_DATA(parent_seq);
    groupidx_ptr = PyArray_DATA(groupidx_seq);
    grouporder_ptr = PyArray_DATA(grouporder_seq);
    groupresult_ptr = PyArray_DATA(groupresult_seq);
    threshold_ptr = PyArray_DATA(threshold_seq);

    i = 0;
    while(i < prob_seq_size)
    {
        max_ptr_cur = array_ptr + MIN(max_ptr - array_ptr,prob_seq_size - i);
        while(array_ptr < max_ptr_cur)
        {
            count_ptr[i] = (uint64_t)((prob_ptr[i] * ((float)(n64 * 64.0))) + (((float)((uint8_t)(*(array_ptr++)))) * ((float)(1.0/255.1))));
            i++;
        }
        if(array_ptr >= max_ptr)
        {
            fill_array64((uint64_t *)array, (((array_ptr - array)/sizeof(double)) + 2) & ~0x1); \
            array_ptr = array;  
        }
    }
    
    for(i = 0; i < parent_seq_size; i++)
        for(j=0; j < n64; j++)
           for(k = 0; k < 64; k++) 
               grouporder_ptr[(i * (n64 * 64)) + (j * 64) + k] = (k << 2) | j;

    double s, ls, pval;
    uint64_t max_count, last_max_count, last_count, cur_count;
    uint64_t cur_pos, rand_pos, curres[n64], * cur_gorder;
    int64_t parent; 
    int idx;


    for(i = 0; i < parent_seq_size; i++)
    {
        parent = parent_ptr[i];
        s = score_ptr[i];
        
        if(parent < 0)
            ls = 0.0;
        else
        {
            if(parent >= i)
            {
                PyErr_SetString(PyExc_TypeError, "Invalid parent index (should only refer back)"); 
                goto error;
            }
            ls = score_ptr[parent];
        }

        cur_gorder = &(grouporder_ptr[(i * n64 * 64)]);

        if(flags & FHIERARCH)
        {
            last_max_count = (((double)n64) * 64.0 * ls) + 0.5;
            max_count = ((double)n64) * 64.0 * s + 0.5;
            if(parent >= 0)
            {
                if(last_max_count > max_count)
                {
                    PyErr_SetString(PyExc_TypeError, "Parent correlation larger than child correlation"); 
                    goto error;
                }
                memcpy(cur_gorder, &(grouporder_ptr[(parent * n64 * 64)]), sizeof(uint64_t) * n64 * 64);
            }
        }
        else
        {
            if(parent >= 0)
            {
                PyErr_SetString(PyExc_TypeError, "No hierarchies allowed with normal score. Use hieararchical scoring function."); 
                goto error;
            }
            last_max_count = 0;
            max_count = n64 * 64;
        }
        if(array_ptr > max_ptr) \
        { \
            if(array_ptr > (array + random_array_size * sizeof(double)))
            {
                printf("Array bounds violated by %lu!!\n", array_ptr - array - random_array_size * sizeof(double));
                return NULL;
            }
            fill_array64((uint64_t *)array, (((array_ptr - array)/sizeof(double)) + 2) & ~0x1); \
            array_ptr = array; \
        }\
         
        rand_pos = 0;
        for(j = 0; j < n64; j++) curres[j] = 0;
        for(j = last_max_count; j < max_count; j++)
        {
            rand_pos = (rand_pos + *(array_ptr++)) % (n64 * 64 - j);
            cur_pos = cur_gorder[rand_pos];
            cur_gorder[rand_pos] = cur_gorder[n64 * 64 - j - 1];
            cur_gorder[n64 * 64 -j - 1] = cur_pos;

            idx = cur_pos & n64_mask;
            curres[idx] |= ((uint64_t) 1) << (cur_pos >> 2);
           
            for(k = 0; k < n64; k++) threshold_ptr[j * n64 + k] = curres[k];
        }
        
        if(parent < 0)
            memcpy(&(groupresult_ptr[(i * 64 * n64 * n64)]), threshold_ptr,sizeof(uint64_t) * n64 * 64 * n64);
        else
        {
            for(j = 1; j <= max_count; j++)
            {
                pval = MIN(((double)j) / (((double)n64) * 64.0 * s), 1.0);
                
                last_count = (uint64_t) (pval * ls * ((double)n64) * 64.0 + 0.5);
                if((j - last_count) > (max_count - last_max_count))
                    last_count += (j - last_count) - (max_count - last_max_count);

                if(last_count == 0)
                    memcpy(&(groupresult_ptr[(i * 64 * n64 * n64) + (j-1) * n64]), &(threshold_ptr[(last_max_count + (j-1)) *n64]) ,sizeof(uint64_t) * n64);
                else
                {
                    cur_count = j - last_count;
                    if(cur_count == 0)
                        memcpy(&(groupresult_ptr[(i * 64 * n64 * n64 + (j-1) * n64)]), &(groupresult_ptr[(parent * 64 * n64 * n64 + (last_count - 1) * n64)]),sizeof(uint64_t) * n64);
                    else
                        for(k = 0; k < n64; k++)
                        {
                            groupresult_ptr[(i * 64 * n64 * n64) + (j-1) * n64 + k] = groupresult_ptr[(parent * 64 * n64 * n64) + (last_count-1) * n64 + k] | threshold_ptr[(last_max_count + cur_count - 1) * n64 + k];
                        }
                }
            }
        }
    }

    int64_t group;
    uint64_t count_done;
    for(i = 0; i < prob_seq_size; i++)
    {
        group = groupidx_ptr[i];
        if(group < 0)
        {
            count_done_ptr[i] = 0;
            memset(&(result_ptr[i * n64]), 0, sizeof(uint64_t) * n64);
        }
        else
        {
            s = score_ptr[group];
            pval = prob_ptr[i];
            if(group > parent_seq_size)
            {
                PyErr_SetString(PyExc_TypeError, "Invalid group index.");
                goto error;
            }
            if(flags & FHIERARCH)
            {
               count_done = MIN((uint64_t)(s * pval * (((double)n64) * 64.0) + 0.5), count_ptr[i]);
            }
            else
            {
               count_done = MIN((uint64_t) (((s * pval * (((double)n64) * 64.0)) / (1.0 - pval + s * pval)) + 0.5),count_ptr[i]);
            }
            count_done_ptr[i] = count_done;
            if(count_done > 0)
                memcpy(&(result_ptr[i * n64]), &(groupresult_ptr[(group * 64 * n64 * n64) + (count_done - 1) * n64]),sizeof(uint64_t) * n64);
            else
                memset(&(result_ptr[i * n64]), 0, sizeof(uint64_t) * n64);
        }
    }

    Py_DECREF(threshold_seq);
    Py_DECREF(groupresult_seq);
    return Py_BuildValue("NNNN",count_seq,count_done_seq,result, grouporder_seq);
error:
    Py_XDECREF(threshold_seq);
    Py_XDECREF(grouporder_seq);
    Py_XDECREF(groupresult_seq);
    Py_XDECREF(count_done_seq);
    Py_XDECREF(count_seq);
    Py_XDECREF(result);
    return NULL;
}



static PyObject * bit_count(PyObject *self, PyObject *count_seq)
{
    PyArray_Descr * uint8descr;
    PyObject * result;
    uint8_t *result_ptr;
    int count_seq_size;
    uint64_t *count_seq_ptr;
    

    if(!PyArray_Check(count_seq) || !PyArray_ISINTEGER(count_seq) || 
       !PyArray_ISUNSIGNED(count_seq) || ((PyArrayObject *)count_seq)->nd != 1 || 
       !PyArray_ISCONTIGUOUS(count_seq) || ((PyArrayObject *)count_seq)->descr->elsize != 8)
    {
        PyErr_SetString(PyExc_TypeError, "Input should be a contiguous 1-dimensional numpy array of unsigned 64bit integers.");
        return NULL;
    }
    count_seq_size = PyArray_SIZE(count_seq);

    uint8descr = PyArray_DescrFromType(NPY_UBYTE);
    result = PyArray_NewFromDescr(&PyArray_Type, uint8descr, 1, PyArray_DIMS(count_seq), NULL, NULL, 0, NULL);
    if(result == NULL)
        return NULL;

    result_ptr = PyArray_DATA(result);
    count_seq_ptr = PyArray_DATA(count_seq);
    while(count_seq_size--)
    {
#if defined(__GNUC__) && (__GNUC__ > 3)
        *result_ptr++ = __builtin_popcountll(*count_seq_ptr++);
#else
        *result_ptr++ = popcount_wegner(*count_seq_ptr++);
#endif
    };
    return result; 
}


static PyObject * bitor_reduceat(PyObject *self, PyObject *args)
{
    PyArray_Descr * uint64descr;
    PyObject * result;
    PyObject *seq, *indexes;
    int seq_size, index_size;
    
    uint64_t *result_ptr, *seq_ptr_base, *seq_ptr_start, *seq_ptr_end;
    uint32_t *index_ptr, *index_end;
    
    if(!PyArg_ParseTuple(args,"OO",&seq,&indexes))
        return NULL;

    if(!PyArray_Check(seq) || !PyArray_ISINTEGER(seq) || 
       !PyArray_ISUNSIGNED(seq) || ((PyArrayObject *)seq)->nd != 1 || 
       !PyArray_ISCONTIGUOUS(seq) || ((PyArrayObject *)seq)->descr->elsize != 8)
    {
        PyErr_SetString(PyExc_TypeError, "Sequence should be a contiguous 1-dimensional numpy array of unsigned 64bit integers.");
        return NULL;
    }
    if(!PyArray_Check(indexes) || !PyArray_ISINTEGER(indexes) || 
       !PyArray_ISUNSIGNED(indexes) || ((PyArrayObject *)indexes)->nd != 1 || 
       !PyArray_ISCONTIGUOUS(indexes) || ((PyArrayObject *)indexes)->descr->elsize != 4)
    {
        PyErr_SetString(PyExc_TypeError, "Indexes should be a contiguous 1-dimensional numpy array of unsigned 32bit integers.");
        return NULL;
    }
    seq_size = PyArray_SIZE(seq);
    index_size = PyArray_SIZE(indexes);

    uint64descr = PyArray_DescrFromType(NPY_ULONGLONG);
    result = PyArray_NewFromDescr(&PyArray_Type, uint64descr, 1, PyArray_DIMS(indexes), NULL, NULL, 0, NULL);
    if(result == NULL)
        return NULL;
    
    if(index_size == 0)
        return result;

    result_ptr = PyArray_DATA(result);
    seq_ptr_base = PyArray_DATA(seq);
    index_ptr = PyArray_DATA(indexes);
    
    if(*index_ptr < 0 || *index_ptr > seq_size)
    {
        PyErr_SetString(PyExc_TypeError, "Index should be between 0 and len(seq).");
        Py_DECREF(result);
        return NULL;
    }

    index_end = index_ptr + index_size;
    seq_ptr_start = seq_ptr_base + *index_ptr++;
    while(index_ptr < index_end)
    {
        if(*index_ptr > seq_size)
        {
            PyErr_SetString(PyExc_TypeError, "Index should be between 0 and len(seq).");
            Py_DECREF(result);
            return NULL;
        }

        seq_ptr_end = seq_ptr_base + *index_ptr++;
        if(seq_ptr_end > seq_ptr_start)
        {
            *result_ptr = *seq_ptr_start++;
            while(seq_ptr_end > seq_ptr_start)
                *result_ptr |= *seq_ptr_start++;
        }
        else
            *result_ptr = 0;
        result_ptr++;
    };
    seq_ptr_end = seq_ptr_base + seq_size;
    *result_ptr = 0;
    while(seq_ptr_end > seq_ptr_start)
        *result_ptr |= *seq_ptr_start++;


    return result; 
}

typedef struct {
    PyObject_HEAD
    double score;
    PyObject * nodein;
    PyObject * nodeout;
    PyObject * epos;
} EdgeObject;

static void
EdgeObject_dealloc(EdgeObject* self)
{
    Py_DECREF(self->nodein);
    Py_DECREF(self->nodeout);
    Py_DECREF(self->epos);
    self->ob_type->tp_free((PyObject*)self);
}

static int
EdgeObject_init(EdgeObject *self, PyObject *args, PyObject *kwds)
{
   if(!PyArg_ParseTuple(args,"OOOd",&self->epos, &(self->nodein),&(self->nodeout), &(self->score)))
       return -1;
   Py_INCREF(self->epos);
   Py_INCREF(self->nodein);
   Py_INCREF(self->nodeout);
   return 0;
}

static PyMethodDef EdgeObject_methods[] = {
    {NULL} 
};

static PyObject *
EdgeObject_getEpos(EdgeObject *self, void *closure)
{
    Py_INCREF(self->epos);
    return self->epos;
}
static PyObject *
EdgeObject_getNodein(EdgeObject *self, void *closure)
{
    Py_INCREF(self->nodein);
    return self->nodein;
}
static PyObject *
EdgeObject_getNodeout(EdgeObject *self, void *closure)
{
    Py_INCREF(self->nodeout);
    return self->nodeout;
}
static PyObject *
EdgeObject_getScore(EdgeObject *self, void *closure)
{
    return PyFloat_FromDouble(self->score);
}
static int
EdgeObject_setEpos(EdgeObject *self, PyObject *value, void *closure)
{
    Py_DECREF(self->epos);
    Py_INCREF(value);
    self->epos = value;
    return 0;
}
static int
EdgeObject_setScore(EdgeObject *self, PyObject *value, void *closure)
{
    if(!PyFloat_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Score should be a float.");
        return -1;
 
    }
    self->score = PyFloat_AsDouble(value);
    return 0;
}


static PyGetSetDef EdgeObject_getseters[] = {
    {"epos", 
     (getter)EdgeObject_getEpos, (setter)EdgeObject_setEpos,
     "edge pos",
     NULL},
    {"nodein", 
     (getter)EdgeObject_getNodein, NULL,
     "Edge node in ",
     NULL},
    {"nodeout", 
     (getter)EdgeObject_getNodeout, NULL,
     "Edge node out ",
     NULL},
    {"score", 
     (getter)EdgeObject_getScore, (setter)EdgeObject_setScore,
     "Edge score ",
     NULL},
    {NULL}  /* Sentinel */
};

int EO_compare(EdgeObject *a, EdgeObject *b)
{
    if(a->ob_type != b->ob_type)
    {
       PyErr_SetString(PyExc_TypeError, "Edge object can only be compared to edge objects");
       return -1;
    }

    return (a->score < b->score) ? -1 : ((a->score == b->score) ? 0 : 1);
}

static PyTypeObject EdgeObject_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "base_container.EdgeObject",             /*tp_name*/
    sizeof(EdgeObject),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)EdgeObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    EO_compare,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,          /*tp_call*/
    0,                         /*tp_str*/
    0,                          /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_CLASS, /*tp_flags*/
    "EdgeObject object",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,             /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    EdgeObject_methods,         /* tp_methods */
    0,                         /* tp_members */
    EdgeObject_getseters,                     /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)EdgeObject_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};/*}}}*/


static PyMethodDef module_methods[] = {
    {"transpose", (PyCFunction) transpose, METH_O, "Transposes nested sequence"},
    {"darray", (PyCFunction) numpy_dimarray, METH_VARARGS, "Constructor for numpy arrays, with min/max dim support."},
    {"random_bits_corr", (PyCFunction) random_bits_corr, METH_VARARGS, "Generates an numpy array with uint64 values, containing n randomly set bits."},
    {"random_bits_input", (PyCFunction) random_bits_input64, METH_VARARGS, "Generates an numpy array with uint64 values, containing n randomly set bits."},
    {"random_bits_input128", (PyCFunction) random_bits_input128, METH_VARARGS, "Generates an numpy array with uint64 values, containing n randomly set bits."},
    {"random_bits_input256", (PyCFunction) random_bits_input256, METH_VARARGS, "Generates an numpy array with uint64 values, containing n randomly set bits."},
    {"random_bits", (PyCFunction) random_bits64, METH_O, "Generates an numpy array with uint64 values, containing n randomly set bits."},
    {"random_bits128", (PyCFunction) random_bits128, METH_O, "Generates an numpy array with uint64 values, containing n randomly set bits."},
    {"random_bits256", (PyCFunction) random_bits256, METH_O, "Generates an numpy array with uint64 values, containing n randomly set bits."},
    {"bit_count", (PyCFunction) bit_count, METH_O, "Count bits for reach element in a 64-bit unsigned integer  numpy array."},
    {"bitor_reduceat", (PyCFunction) bitor_reduceat, METH_VARARGS, "Reduceats an 64 uint numpy array based on index."},
    {NULL}  /* Sentinel */
};



#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initcutils(void) 
{
    PyObject* m;
    EdgeObject_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&EdgeObject_Type) < 0)
        return;
    
    m = Py_InitModule3("cutils", module_methods,
                       "C utility functions");
    Py_INCREF(&EdgeObject_Type);
    PyModule_AddObject(m, "Edge", (PyObject *)&EdgeObject_Type);

    import_array();
    PyObject * globals = PyEval_GetGlobals();

    random_module = PyImport_ImportModuleEx("random", globals, NULL, NULL);
    if(random_module == NULL)
        return;
    

    return;

}



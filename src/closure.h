#ifndef Py_CLOSURE_H
#define Py_CLOSURE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    PyObject_HEAD
    PyObject * obj_weak;
    PyObject * obj_dict;
} PyDWBaseObject;


#define PyDescr_NewClosureMethod_NUM 0
#define PyDescr_NewClosureMethod_RETURN PyObject *
#define PyDescr_NewClosureMethod_PROTO (PyTypeObject *, PyMethodDef *, PyObject *)

#define PyDescr_NewClosureClassMethod_NUM 1
#define PyDescr_NewClosureClassMethod_RETURN PyObject *
#define PyDescr_NewClosureClassMethod_PROTO (PyTypeObject *, PyMethodDef *, PyObject *)

#define PyClosureFunction_New_NUM 2
#define PyClosureFunction_New_RETURN PyObject *
#define PyClosureFunction_New_PROTO (PyMethodDef *, PyObject *, PyObject *, PyObject *)

#define PyDWBaseObject_Type_NUM 3

/* Total number of C API pointers */
#define PyClosure_API_pointers 4 


#ifdef CLOSURE_MODULE
    static PyDescr_NewClosureMethod_RETURN PyDescr_NewClosureMethod PyDescr_NewClosureMethod_PROTO;
    static PyDescr_NewClosureClassMethod_RETURN PyDescr_NewClosureClassMethod PyDescr_NewClosureClassMethod_PROTO;
    static PyClosureFunction_New_RETURN PyClosureFunction_New PyClosureFunction_New_PROTO;

#else
    static void **PyClosure_API;
    #define PyDWBaseObject_Type \
        ((PyTypeObject *) PyClosure_API[PyDWBaseObject_Type_NUM])

    #define PyDescr_NewClosureMethod \
    (*(PyDescr_NewClosureMethod_RETURN (*)PyDescr_NewClosureMethod_PROTO) PyClosure_API[PyDescr_NewClosureMethod_NUM])
    
    #define PyDescr_NewClosureClassMethod \
    (*(PyDescr_NewClosureClassMethod_RETURN (*)PyDescr_NewClosureClassMethod_PROTO) PyClosure_API[PyDescr_NewClosureClassMethod_NUM])

    #define PyClosureFunction_New \
    (*(PyClosureFunction_New_RETURN (*)PyClosureFunction_New_PROTO) PyClosure_API[PyClosureFunction_New_NUM])

    /* Return -1 and set exception on error, 0 on success. */
    static int
    import_closure(void)
    {
        PyObject *globals = PyEval_GetGlobals();
        PyObject *module = PyImport_ImportModuleEx("closure",globals,NULL,NULL);

        if (module != NULL) 
        {
            PyObject *c_api_object = PyObject_GetAttrString(module, "_C_API");
            if (c_api_object == NULL)
                return -1;
            if (PyCObject_Check(c_api_object))
                PyClosure_API = (void **)PyCObject_AsVoidPtr(c_api_object);
            else
                return -1;
            Py_DECREF(c_api_object);
        }
        else
            return -1;
        return 0;
    }
#endif


#ifdef __cplusplus
}
#endif

#endif


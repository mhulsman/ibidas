#ifndef Py_BCONTAINER_H
#define Py_BCONTAINER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    PyObject_HEAD
    long nrmod;
    long curid;
} ModificationSemaphore;

typedef struct {
    PyObject_HEAD
    PyObject * data;
    PyObject * shape;
    PyObject * fields;
    PyObject * modify_check;
    long sem_state;
    char cacheable; //should be set to true if there is no chance that this result set will change (in any of the dependent querys)
} QueryResult;

typedef struct {
    PyObject_HEAD
    PyObject * descriptors;
    PyObject * actidx_descriptors;
    PyObject * fields;
    PyObject * actidx;
    PyObject * dims;
    PyObject * actidx_dims;
    PyObject * segments;
    PyObject * modify_sems;
    PyObject * select_query;
    PyObject * cache;
    PyObject * cap_class;
    PyObject * local;

} Invar;

typedef struct {
PyObject_HEAD
    PyObject * init;
    PyObject * source_invar;
    PyObject * key;
} RegisterContext;


typedef struct {
    PyObject_HEAD
    Invar * invar;         
    PyObject * result;        
    PyObject * source;
} BaseContainer;


#define PyBContainer_API_pointers 1 

#define QueryResult_copy_NUM 0
#define QueryResult_copy_RETURN PyObject *
#define QueryResult_copy_PROTO (QueryResult *)


#ifdef BASECONTAINER_MODULE
    static PyTypeObject ModificationSemaphore_Type;
    static PyTypeObject QueryResult_Type;
    static PyTypeObject BaseContainer_Type;
    static PyTypeObject Invar_Type;
    static PyTypeObject RegisterContext_Type;
    
    static PyObject * QueryResult_copy(QueryResult *self);
#else
    static void **PyBContainer_API;

    #define QueryResult_copy \
        (*(QueryResult_copy_RETURN (*)QueryResult_copy_PROTO) PyBContainer_API[QueryResult_copy_NUM])

    /* This section is used in modules that use spammodule's API */
    static PyTypeObject * ModificationSemaphore_Type = NULL;
    static PyTypeObject * QueryResult_Type = NULL;
    static PyTypeObject * BaseContainer_Type = NULL;
    static PyTypeObject * Invar_Type = NULL;
        
    /* Return -1 and set exception on error, 0 on success. */
    static int
    import_base_container(void)
    {
        PyObject *globals = PyEval_GetGlobals();
        PyObject *module = PyImport_ImportModuleEx("base_container",globals,NULL,NULL);

        if (module != NULL) 
        {
            PyObject *c_api_object = PyObject_GetAttrString(module, "_C_API");
            if (c_api_object == NULL)
                return -1;
            if (PyCObject_Check(c_api_object))
                PyBContainer_API = (void **)PyCObject_AsVoidPtr(c_api_object);
            Py_DECREF(c_api_object);

            ModificationSemaphore_Type = (PyTypeObject *)PyObject_GetAttrString(module, "ModificationSemaphore");
            QueryResult_Type = (PyTypeObject *)PyObject_GetAttrString(module, "QueryResult");
            BaseContainer_Type = (PyTypeObject *)PyObject_GetAttrString(module, "BaseContainer");
            Invar_Type = (PyTypeObject *)PyObject_GetAttrString(module, "Invar");
            
        }
        return 0;
    }
#endif


#ifdef __cplusplus
}
#endif

#endif


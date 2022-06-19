#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/arrayobject.h"
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

static char dominates(npy_double *dataa, npy_double *datab, npy_intp stride, npy_intp count) {
    char any, all;
    any = 0;
    all = 1;

    while (count--) {
        if (*dataa < *datab) {
            all = 0;
            break;
        } else if (*dataa > *datab) {
            any = 1;
        }
        dataa += stride;
        datab += stride;
    }    

    return any && all;
}

static PyObject *dominationScores(PyObject *self, PyObject *args) {
    // Error return value
    PyObject *errVal = PyLong_FromLong(-1);

    // Input args
    PyObject *arg1;

    // Array
    PyArrayObject *arr;

    // Array shape
    int dims;
    npy_intp dim0count, dim1count;

    // Array strides
    npy_intp dim0stride, dim1stride;

    // Array data pointer
    npy_double *data;

    // Parse arguments any create PyArrayObject
    if (!PyArg_ParseTuple(args, "O", &arg1)) {
        return errVal;
    }
    arr = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_DOUBLE,  NPY_ARRAY_IN_ARRAY);
    if (arr == NULL) {
        return errVal;
    }

    // Get shape, and check array is 2D
    dims = PyArray_NDIM(arr);
    if (!(dims == 2)) {
        return errVal;
    }
    dim0count = PyArray_DIM(arr, 0);
    dim1count = PyArray_DIM(arr, 1);

    // Get stride values
    dim0stride = PyArray_STRIDE(arr, 0)/sizeof(npy_double);
    dim1stride = PyArray_STRIDE(arr, 1)/sizeof(npy_double);

    // Get data pointer
    data = (npy_double *)PyArray_BYTES(arr);

    // Create output array
    int outputDims[1] = { dim0count };
    PyArrayObject *output = (PyArrayObject *)PyArray_FromDims(1, outputDims, NPY_INT);

    // Create dominationMatrix, strengths array and domination scores array
    npy_bool dominationMatrix[dim0count * dim0count];
    memset(dominationMatrix, 0, sizeof(dominationMatrix));
    npy_int strengths[dim0count];
    memset(strengths, 0, sizeof(strengths));
    npy_int *dominationScores = (npy_int *)output->data;

    // Create pointers for score vectors and allocate loop counter variables
    npy_double *scores1 = data;
    npy_double *scores2;
    npy_intp i, j;

    for (i = 0; i < dim0count; i++) {
        scores2 = scores1 + dim0stride;
        for (j = i + 1; j < dim0count; j++) {
            if (dominates(scores1, scores2, dim1stride, dim1count)) {
                strengths[i] += 1;
                dominationMatrix[i + j * dim0count] = 1;
            } else if (dominates(scores2, scores1, dim1stride, dim1count)) {
                strengths[j] += 1;
                dominationMatrix[j + i * dim0count] = 1;
            } 
            scores2 += dim0stride;
        }
        scores1 += dim0stride;
    }

    // Allocate domination score variable
    npy_int dominationScore;

    for (i = 0; i < dim0count; i++) {
        dominationScore = 0;
        for (j = 0; j < dim0count; j++) {
            if (dominationMatrix[j + i * dim0count]) {
                dominationScore += strengths[j];
            }
        }
        dominationScores[i] = dominationScore;
    }
    
    Py_DECREF(arr);
    
    return (PyObject *)output;
}

static PyMethodDef spea2dominationscoresMethods[] = {
    {"domination_scores", dominationScores, METH_VARARGS,
     "Calculate population domination scores from population objective scores."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef spea2dominationscores = {
   PyModuleDef_HEAD_INIT,
   "spea2domnationscores",
   NULL,
   -1,
   spea2dominationscoresMethods
};

PyMODINIT_FUNC
PyInit_spea2dominationscores(void) {
    import_array();
    return PyModule_Create(&spea2dominationscores);
}

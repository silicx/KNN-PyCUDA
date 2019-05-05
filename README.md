# KNN-PyCUDA
KNN implementation with PyCUDA


    KNN_pycuda(MAX_K, X_train, X_test, y_train, y_test,
               metric='eucl', preproc=None, verbose=False):
    
parameters:
- `MAX_K`: the function evaluates every K in [1, MAX_K]
- `X_train`, X_test, y_train, y_test: training and test set
- `metric`: distance metric. {'eucl', 'manh', 'cheb', 'cos'} are supported
- `preproc`: pre-processing method (normalization). {None, 'l1', 'l2', 'zscore'} are supported
- `verbose`: displays progress if verbose=True

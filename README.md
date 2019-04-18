# KNN-PyCUDA
KNN implementation with PyCUDA


KNN_pycuda(MAX_K, X_train, X_test, y_train, y_test,
               metric='eucl', preproc=None, verbose=False):
    
- parameters:
- MAX_K: the function evaluate every K in [1, MAX_K]
- X_train, X_test, y_train, y_test: training and test set
- metric: distance metric. Support 'eucl', 'manh', 'cheb', 'cos'
- preproc: pre-process (normalization). Support None, 'l1', 'l2', 'zscore'
- verbose: display progress if verbose=True

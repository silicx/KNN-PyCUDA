from __future__ import print_function

import os

import numpy as np

import pycuda.driver as driver
import pycuda.autoinit
from pycuda.compiler import SourceModule

from time import strftime, localtime
from datetime import datetime, timedelta



def timestamp(info="", offset=0):
    tm = datetime.now()+timedelta(hours=offset)
    print(tm.time().strftime("%H:%M:%S"), info)



def KNN_pycuda(MAX_K, X_train, X_test, y_train, y_test, metric='eucl', preproc=None, verbose=False):
    """
    KNN_pycuda(MAX_K, X_train, X_test, y_train, y_test,
               metric='eucl', preproc=None, verbose=False):
    
    parameters:

    MAX_K: the function evaluate every K in [1, MAX_K]
    X_train, X_test, y_train, y_test: training and test set
    metric: distance metric. Support 'eucl', 'manh', 'cheb', 'cos'
    preproc: pre-process (normalization). Support None, 'l1', 'l2', 'zscore'
    verbose: display progress if verbose=True
    """


    assert metric in ['manh', 'eucl', 'cheb', 'cos']
    assert preproc in [None, 'zscore', 'l2', 'l1']
    assert MAX_K < X_train.shape[0]
    
    if preproc=='zscore':
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train = (X_train-mean)/std
        X_test = (X_test-mean)/std
    elif preproc == 'l2':
        X_train = np.transpose( X_train.transpose() / np.sqrt(np.sum(np.square(X_train), axis=1)) )
        X_test  = np.transpose( X_test.transpose() / np.sqrt(np.sum(np.square(X_test), axis=1)) )
    elif preproc == 'l1':
        X_train = np.transpose( X_train.transpose() / np.sum(np.abs(X_train), axis=1) )
        X_test  = np.transpose( X_test.transpose() / np.sum(np.abs(X_test), axis=1) )
    
    
    BSIZE = 32
    
    Nclass = 1 + max(y_train.max(), y_test.max())
    Ntest = X_test.shape[0]
    Ntrain = X_train.shape[0]
    Dim = X_train.shape[1]
    
    
    
    # declare and compile kernel
    mod = SourceModule("""
    #include <cmath>
    using namespace std;
    
    #define Ntrain """+str(Ntrain)+"""
    #define Ntest """+str(Ntest)+"""
    #define Nclass """+str(Nclass)+"""
    #define Dim """+str(Dim)+"""
    #define BSIZE """+str(BSIZE)+"""
    #define MAX_K """+str(MAX_K)+"""
    
    
    __global__ void cuda_l2(float *dest, float *A, float *B) 
    {
        /* (m,d)*(n,d) -> (n*m)  */
        
        const int ind_x = threadIdx.x + blockIdx.x*blockDim.x;
        const int ind_y = threadIdx.y + blockIdx.y*blockDim.y;
        
        const int stride_x = blockDim.x * gridDim.x;
        const int stride_y = blockDim.y * gridDim.y;
        
        
        for(int x = ind_x; x<Ntrain; x += stride_x){
            for(int y = ind_y; y<Ntest; y += stride_y){
                float *res = dest + y*Ntrain + x;
                float *src_a = A + x*Dim;
                float *src_b = B + y*Dim;
                
                float t;
                
                *res = 0;
                for(int k=0; k<Dim; ++k, ++src_a, ++src_b){
                    t = *src_a - *src_b;
                    *res += t*t;
                }
                *res = sqrt(*res);
            }
        }
    }
    
    __global__ void cuda_l1(float *dest, float *A, float *B) 
    {
        /* (m,d)*(n,d) -> (n*m)  */
        
        const int ind_x = threadIdx.x + blockIdx.x*blockDim.x;
        const int ind_y = threadIdx.y + blockIdx.y*blockDim.y;
        
        const int stride_x = blockDim.x * gridDim.x;
        const int stride_y = blockDim.y * gridDim.y;
        
        
        for(int x = ind_x; x<Ntrain; x += stride_x){
            for(int y = ind_y; y<Ntest; y += stride_y){
                float *res = dest + y*Ntrain + x;
                float *src_a = A + x*Dim;
                float *src_b = B + y*Dim;
                
                float t;
                
                *res = 0;
                
                for(int k=0; k<Dim; ++k, ++src_a, ++src_b){
                    t = *src_a - *src_b;
                    *res += (t>0?t:-t);
                }
            }
        }
    }
    
    __global__ void cuda_loo(float *dest, float *A, float *B) 
    {
        /* (m,d)*(n,d) -> (n*m)  */
        
        const int ind_x = threadIdx.x + blockIdx.x*blockDim.x;
        const int ind_y = threadIdx.y + blockIdx.y*blockDim.y;
        
        const int stride_x = blockDim.x * gridDim.x;
        const int stride_y = blockDim.y * gridDim.y;
        
        
        for(int x = ind_x; x<Ntrain; x += stride_x){
            for(int y = ind_y; y<Ntest; y += stride_y){
                float *res = dest + y*Ntrain + x;
                float *src_a = A + x*Dim;
                float *src_b = B + y*Dim;
                
                float t;
                
                *res = 0;
                
                for(int k=0; k<Dim; ++k, ++src_a, ++src_b){
                    t = *src_a - *src_b;
                    t = (t>0?t:-t);
                    if(t > *res)
                        *res = t;
                }
            }
        }
    }
    
    __global__ void cuda_cos(float *dest, float *A, float *B) 
    {
        /* (m,d)*(n,d) -> (n*m)  */
        
        const int ind_x = threadIdx.x + blockIdx.x*blockDim.x;
        const int ind_y = threadIdx.y + blockIdx.y*blockDim.y;
        
        const int stride_x = blockDim.x * gridDim.x;
        const int stride_y = blockDim.y * gridDim.y;
        
        
        for(int x = ind_x; x<Ntrain; x += stride_x){
            for(int y = ind_y; y<Ntest; y += stride_y){
                float *res = dest + y*Ntrain + x;
                float *src_a = A + x*Dim;
                float *src_b = B + y*Dim;
                
                float t1=0, t2=0;
                
                *res = 0;
                
                for(int k=0; k<Dim; ++k, ++src_a, ++src_b){
                    *res += (*src_a) * (*src_b);
                    t1 += (*src_a) * (*src_a);
                    t2 += (*src_b) * (*src_b);
                }
                
                *res /= sqrt(*src_a) * sqrt(*src_b);
            }
        }
    }
    
    
    
    __device__ void heap_movedown(int *index, float *dist, int h, int len){
        int xi = index[h];
        float xd = dist[xi];

        int t = h<<1|1;
        int *cur_index;
        
        while(t<len){
            cur_index = index + t;
            
            if(t+1<len && dist[*(cur_index+1)] < dist[*cur_index]){
                ++t;
                ++cur_index;
            }
            
            if(dist[*cur_index] < xd){
                index[h] = *cur_index;

                h = t;
                t = h<<1|1;
            }else break;
        }

        index[h] = xi;
    }

    __device__ void partial_heapsort(int *index, float *dist, int len){
        for(int i=(len>>1); i>=0; --i)
            heap_movedown(index, dist, i, len);

        for(int i=len-1;i>=len-MAX_K;i--){
            int s = *(index);
            *(index) = *(index+i);
            *(index+i) = s;

            heap_movedown(index, dist, 0, i);
        }
    }
    
    
    __global__ void cuda_argsort(int *index, float *distance)
    {
        /* (n*m) sorted by axis=1*/
        
        
        const int ind_x = threadIdx.x + blockIdx.x*blockDim.x;
        const int ind_y = threadIdx.y + blockIdx.y*blockDim.y;
        
        const int stride_x = blockDim.x * gridDim.x;
        const int stride_y = blockDim.y * gridDim.y;
        
        
        for(int y = ind_y; y<BSIZE; y += stride_y)
            for(int x = ind_x; (x*BSIZE+y)<Ntest; x += stride_x){
                int id = (x*BSIZE+y)*Ntrain;
                
                int *cur_index = index+id;
                for(int i=0; i<Ntrain; ++i, ++cur_index)
                    *cur_index = i;

                partial_heapsort(index+id, distance+id, Ntrain);
            }
    }
    
    
    
    __global__ void cuda_vote(int *acc, int *index, int *y_train, int *y_test)
    {
        const int ind_x = threadIdx.x + blockIdx.x*blockDim.x;
        const int ind_y = threadIdx.y + blockIdx.y*blockDim.y;
        
        const int stride_x = blockDim.x * gridDim.x;
        const int stride_y = blockDim.y * gridDim.y;
        
        
        int *counter = new int[Nclass];
        
        for(int y = ind_y; y<BSIZE; y += stride_y) 
            for(int x = ind_x; (x*BSIZE+y)<Ntest; x += stride_x){
                int id = (x*BSIZE+y);
                
                for(int j=0;j<Nclass;j++)
                    counter[j] = 0;
                    
                int *cur_index = index + (id+1)*Ntrain-1;
                int *cur_acc = acc + id*MAX_K;
                
                for(int k=0; k<MAX_K; ++k, --cur_index, ++cur_acc){
                    counter[ y_train[*cur_index] ] += 1;
                    
                    int *maxi = counter;
                    for(int j=1;j<Nclass;j++)
                        if(*(counter+j) > *maxi)
                            maxi = counter+j;
                            
                    *cur_acc = ((maxi-counter) == y_test[id] ? 1 : 0);
                }
            }
            
            
        delete [] counter;
    }
    
    """)
    
    
    cuda_argsort = mod.get_function("cuda_argsort")
    cuda_vote    = mod.get_function("cuda_vote")
    
    
    if metric=='eucl':
        cuda_metric  = mod.get_function("cuda_l2")
    elif metric=='manh':
        cuda_metric  = mod.get_function("cuda_l1")
    elif metric=='cheb':
        cuda_metric  = mod.get_function("cuda_loo")
    elif metric=='cos':
        cuda_metric  = mod.get_function("cuda_cos")
    
    
    cuda_metric = cuda_metric.prepare("PPP")
    cuda_argsort = cuda_argsort.prepare("PP")
    cuda_vote = cuda_vote.prepare("PPPP")
    
    
    
    
    if verbose: timestamp("Start measuring distance") ###########################################################################################


    
    # prepare data, send to GPU
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    X_train_gpu = driver.mem_alloc(X_train.nbytes)
    driver.memcpy_htod(X_train_gpu, X_train)
    X_test_gpu = driver.mem_alloc(X_test.nbytes)
    driver.memcpy_htod(X_test_gpu, X_test)
    
    
    dist = np.empty((Ntest, Ntrain), dtype=np.float32)
    dist_gpu = driver.mem_alloc(dist.nbytes)
    
    
    # run
    cuda_metric.prepared_call((Ntrain//BSIZE+1, Ntest//BSIZE+1),
                              (BSIZE, BSIZE, 1),
                              dist_gpu, X_train_gpu, X_test_gpu)
    
    
    # retrieve
    if verbose: driver.memcpy_dtoh(dist, dist_gpu)
    
    
    
    if verbose: timestamp("Finish measuring distance, Start sorting") ################################################################################
    
    
    # prepare
    indsort = np.empty((Ntest, Ntrain), dtype=np.int32)
    indsort_gpu = driver.mem_alloc(indsort.nbytes)
    
    
    
    # run
    cuda_argsort.prepared_call((Ntest//(BSIZE*BSIZE)+1, 1),
                               (BSIZE,BSIZE,1),
                               indsort_gpu, dist_gpu)
    
    
    # retieve
    if verbose: driver.memcpy_dtoh(indsort, indsort_gpu)
    
    
    
    
    if verbose: timestamp("Finish sorting, start voting") ###########################################################################################
    
    
    # prepare
    acc = np.empty((Ntest, MAX_K), dtype=np.int32)
    acc_gpu = driver.mem_alloc(acc.nbytes)
    
    y_test = y_test.astype(np.int32)
    y_test_gpu = driver.mem_alloc(y_test.nbytes)
    driver.memcpy_htod(y_test_gpu, y_test)
    
    y_train = y_train.astype(np.int32)
    y_train_gpu = driver.mem_alloc(y_train.nbytes)
    driver.memcpy_htod(y_train_gpu, y_train)
    
    # run
    cuda_vote.prepared_call((Ntest//(BSIZE*BSIZE)+1, 1),
                            (BSIZE,BSIZE,1),
                            acc_gpu, indsort_gpu, y_train_gpu, y_test_gpu)
    
    
    
    # retrive
    
    driver.memcpy_dtoh(acc, acc_gpu)
    
    
    
    acc = acc.mean(axis=0)
    
    
    
    if verbose: timestamp("Release memory") #########################################################################################################
    
    print("Best accuracy =", acc.max(), "K =", acc.argmax())
    
    
    X_train_gpu.free()
    X_test_gpu.free()
    
    dist_gpu.free()
    indsort_gpu.free()
    
    y_test_gpu.free()
    acc_gpu.free()
    
    if verbose: timestamp("Done.") #########################################################################################################
    

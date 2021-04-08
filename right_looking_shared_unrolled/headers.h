#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<cmath>
__device__ void store_full_row(float*,float*,int,int);
__device__ void load_full_row(float*,float*,int,int);
__device__ void store_full(float*,float*,int,int,int);
__device__ void load_full(float*,float*,int,int,int);
__device__ void store_lower(float*,float*,int,int,int);
__device__ void load_lower(float*,float*,int,int,int);
__device__ void potrf_tile(float*);
__device__ void trsm_tile(float*,int,int,int);
__device__ void syrk_tile(float*,float*,int,int,int);
__device__ void store_zeros(float*,int);
__global__ void right_looking_launch_kernel(float*,int);
#include "aux_1.cu"
#include "aux_2.cu"
#include "right_looking_kernel_code.cu"
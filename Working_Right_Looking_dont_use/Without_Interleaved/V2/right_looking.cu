#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<cmath>
#define TILE_SIZE 2
__device__ void store_full(float*,float*,int);
__device__ void load_full(float*,float*,int);
__device__ void potrf_tile(float*,int,int);
__device__ void trsm_tile(float*,int,int,int);
__device__ void syrk_tile(float*,int,int,int,int);
__global__ void right_looking_launch_kernel(float*,int);
__device__ void store_zeros(float*,int);
__device__ void store_zeros_diagonal(float*,int,int);
__device__ void store_zeros_last(float*,int);

__device__ void store_full(float* read_data,float* write_data,int N)
{
    int i,j,ID;
    for(i=0;i<N/TILE_SIZE;i++)
    {
        for(j=0;j<N/TILE_SIZE;j++)
        {
            ID = (i*TILE_SIZE + threadIdx.y)*N + j*TILE_SIZE + threadIdx.x;
            write_data[ID + N*N*blockIdx.x] = read_data[ID];
        }
    }
    __syncthreads();
}
__device__ void load_full(float* read_data,float* write_data,int N)
{
    int i,j,ID;
    for(i=0;i<N/TILE_SIZE;i++)
    {
        for(j=0;j<N/TILE_SIZE;j++)
        {
            ID = (i*TILE_SIZE + threadIdx.y)*N + j*TILE_SIZE + threadIdx.x;
            write_data[ID] = read_data[ID + N*N*blockIdx.x];
        }
    }
    __syncthreads();
}
__device__ void potrf_tile(float* t_A,int i,int N)
{
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    for(int k=0;k<TILE_SIZE;k++)
    {
        if(t_x==t_y && t_x==k)
        {
            t_A[i*TILE_SIZE*(1+N) + t_x*N + t_x] = sqrtf(t_A[i*TILE_SIZE*(1+N) + t_x*N + t_x]);
        }
        __syncthreads();
        if(t_x<t_y && t_x == k)
        {
            t_A[i*TILE_SIZE*(1+N) + t_y*N + t_x]/= t_A[i*TILE_SIZE*(1+N) + t_x*N + t_x];
        }
        __syncthreads();
        if(k<t_y && k<t_x && t_x<=t_y)
        {
            t_A[i*TILE_SIZE*(1+N) + t_y*N + t_x]-= t_A[i*TILE_SIZE*(1+N) + t_x*N + k]*t_A[i*TILE_SIZE*(1+N) + t_y*N + k];
        }
        __syncthreads();
    }
}
__device__ void trsm_tile(float *row_data,int i,int j,int N)
{
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    for(int s=0;s<TILE_SIZE;s++)
    {
        if(t_x==s)
        {
            row_data[(t_y + j*TILE_SIZE)*N + t_x + i*TILE_SIZE]/= row_data[i*TILE_SIZE*(1+N) + t_x*(1+N)];
        }
        __syncthreads();
        if(t_x > s)
        {
            row_data[(t_y + j*TILE_SIZE)*N + t_x + i*TILE_SIZE]-= row_data[(t_x + i*TILE_SIZE)*N +  s]*row_data[(t_y + j*TILE_SIZE)*N + s];
        }
        __syncthreads();
    }
}
__device__ void syrk_tile(float* row_data,int i,int j,int k,int N) 
{
    int t_y = threadIdx.y;
    int t_x = threadIdx.x;
    float valueToSubtract = 0.0;
    for(int r=0;r<TILE_SIZE;r++)
    {
        valueToSubtract+= row_data[(t_x + k*TILE_SIZE)*N + i*TILE_SIZE + r]*row_data[(t_y + j*TILE_SIZE)*N + i*TILE_SIZE + r];
    }
    row_data[(t_y + j*TILE_SIZE)*N + t_x + k*TILE_SIZE]-= valueToSubtract;
    __syncthreads();
}
__device__ void store_zeros(float* A,int N)
{
    int t_y = threadIdx.y;
    int t_x = threadIdx.x;
    int i,j;
    for(i=0;i<N/TILE_SIZE-1;i++)
    {
        for(j=i+1;j<N/TILE_SIZE;j++)
            A[j*blockDim.x + t_x + (i*blockDim.y + t_y)*N] = 0.0;
    }
    __syncthreads();
}
__device__ void store_zeros_diagonal(float* A,int N,int b)        // Will only work if (N/TILE_SIZE) is even
{
    int t_y = threadIdx.y;
    int t_x = threadIdx.x;
    int i;
    for(i=0;i<N/TILE_SIZE-b;i+=2)
    {
        if(t_x>t_y)
            A[i*blockDim.x + t_x + (i*blockDim.y + t_y)*N] = 0.0;
        if(t_x<t_y)
            A[(i+1)*blockDim.x*(1+N) + (blockDim.x-t_x-1) + (blockDim.y-t_y-1)*N] = 0.0;
    }
    __syncthreads();
}
__device__ void store_zeros_last(float* A,int N)        // Will only work if (N/TILE_SIZE) is even
{
    int t_y = threadIdx.y;
    int t_x = threadIdx.x;
    if(t_x>t_y)
        A[(N/TILE_SIZE-1)*blockDim.x*(1+N) + t_x + t_y*N] = 0.0;
    __syncthreads();
}
__global__ void right_looking_launch_kernel(float* read_data,int N)
{
    extern __shared__ float data[];
    int i,j,k;
    load_full(read_data,data,N);
    for(i=0;i<N/TILE_SIZE;i++)
    {
        potrf_tile(data,i,N);
        for(j=i+1;j<N/TILE_SIZE;j++)
        {
            trsm_tile(data,i,j,N);
            for(k=i+1;k<=j;k++)
            {
                syrk_tile(data,i,j,k,N);
            }
        }
    }
    store_zeros(data,N);
    if((N/TILE_SIZE)%2==0)
        store_zeros_diagonal(data,N,0);
    else
    {
        store_zeros_diagonal(data,N,1);
        store_zeros_last(data,N);
    }
    store_full(data,read_data,N);
}
#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<cmath>
#define TILE_SIZE 32            //Tile size and block size, both are taken as 32
__device__ void store_full(float*,float*,int,int,int);
__device__ void load_full(float*,float*,int,int,int);
__device__ void store_lower(float*,float*,int,int,int);
__device__ void load_lower(float*,float*,int,int,int);
__device__ void potrf_tile(float*);
__device__ void trsm_tile(float*,int,int,int);
__device__ void syrk_tile(float*,float*,int,int,int);
__global__ void right_looking_launch_kernel(float*,int);

__device__ void store_full(float* read_data,float* write_data,int i,int j,int N)
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = i*blockDim.x + threadIdx.x;
        write_data[global_y*N + global_x] = read_data[threadIdx.x + TILE_SIZE*threadIdx.y];
    __syncthreads();
}
__device__ void load_full(float* read_data,float* write_data,int i,int j,int N)
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = i*blockDim.x + threadIdx.x;
        write_data[threadIdx.x + TILE_SIZE*threadIdx.y] = read_data[global_y*N + global_x];
    __syncthreads();
}
__device__ void store_lower(float* read_data,float* write_data,int i,int j,int N)
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = i*blockDim.x + threadIdx.x;
    if(threadIdx.y >= threadIdx.x)
        write_data[global_y*N + global_x] = read_data[threadIdx.x + TILE_SIZE*threadIdx.y];
    else
        write_data[global_y*N + global_x] = 0.0;
    __syncthreads();
}
__device__ void load_lower(float* read_data,float* write_data,int i,int j,int N)
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = i*blockDim.x + threadIdx.x;
    if(threadIdx.y >= threadIdx.x)
        write_data[threadIdx.x + TILE_SIZE*threadIdx.y] = read_data[global_y*N + global_x];
    else
        write_data[threadIdx.x + TILE_SIZE*threadIdx.y] = 0.0;
    __syncthreads();
}
__device__ void potrf_tile(float* t_A)
{
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    __shared__ float temp2;                        // Using shared memory to Optimize
    for(int k=0;k<TILE_SIZE;k++)
    {
        if(t_x==t_y && t_x==k)
        {
            t_A[k*TILE_SIZE + k] = sqrtf(t_A[k*TILE_SIZE + k]);
            temp2 = t_A[k*TILE_SIZE + k];
        }
        __syncthreads();
        if(t_x<t_y && t_x == k)
        {
            t_A[t_y*TILE_SIZE + k]/= temp2;
        }
        __syncthreads();
        if(k<t_y && k<t_x && t_x<=t_y)
        {
            t_A[t_y*TILE_SIZE + t_x]-= t_A[t_x*TILE_SIZE + k]*t_A[t_y*TILE_SIZE + k];
        }
        __syncthreads();
    }
}
__device__ void trsm_tile(float *read_data,int i,int j,int N)
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = i*blockDim.x + threadIdx.x;
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    for(int s=0;s<TILE_SIZE;s++)
    {
	if(t_x==s)
        {
	    read_data[global_y*N + global_x]/= read_data[global_x*N + global_x];
	}
	__syncthreads();
	if(t_x > s)
        {
	    read_data[global_y*N + global_x]-= read_data[global_x*N + global_x - t_x + s]*read_data[global_y*N + global_x - t_x + s];
	}
	__syncthreads();
    }
}
__device__ void syrk_tile(float* read_data,float* rA2,int i,int j,int k,int N) 
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = k*blockDim.x + threadIdx.x;
    int t_y = threadIdx.y;
    int t_x = threadIdx.x;
    __shared__ float temp0[TILE_SIZE][TILE_SIZE];
    __shared__ float temp1[TILE_SIZE][TILE_SIZE];
    temp0[t_y][t_x] = read_data[global_x*N + i*blockDim.x + t_y];
    temp1[t_x][t_y] = read_data[global_y*N + i*blockDim.x + t_x];
    __syncthreads();
    float valueToSubtract = 0.0;
    for(int r=0;r<TILE_SIZE;r++)
    {
        valueToSubtract+= temp0[r][t_x]*temp1[r][t_y];
    }
    rA2[t_y*TILE_SIZE + t_x]-= valueToSubtract;
    __syncthreads();
}
__global__ void right_looking_launch_kernel(float* read_data,int N)
{
    __shared__ float block_data[TILE_SIZE*TILE_SIZE];
    int i,j,k;
    for(i=0;i<N/TILE_SIZE;i++)
    {
        load_lower(read_data,block_data,i,i,N);
        potrf_tile(block_data);
        store_lower(block_data,read_data,i,i,N);
        for(j=i+1;j<N/TILE_SIZE;j++)
        {
            trsm_tile(read_data,i,j,N);
            for(k=i+1;k<j;k++)
            {
                load_full(read_data,block_data,k,j,N);
                syrk_tile(read_data,block_data,i,j,k,N);
                store_full(block_data,read_data,k,j,N);
            }
            load_lower(read_data,block_data,k,j,N);        
            syrk_tile(read_data,block_data,i,j,k,N);
            store_lower(block_data,read_data,k,j,N);
        }
    }
}
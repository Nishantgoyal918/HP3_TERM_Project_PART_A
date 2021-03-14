#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<cmath>
#define TILE_SIZE 32            //Tile size and block size, both are taken as 32
__device__ void store_full(float*,float*);
__device__ void load_full(float*,float*);
__device__ void store_lower(float*,float*);
__device__ void load_lower(float*,float*);
__device__ void potrf_tile(float*);
__device__ void trsm_tile(float*,float*);
__device__ void syrk_tile_1(float*,float*);
__device__ void syrk_tile_2(float*,float*);
__global__ void right_looking_kernel(float*,int);

__device__ void store_full(float* read_data,float* write_data)
{
    int global_y = blockIdx.y*blockDim.y + threadIdx.y;
    int global_x = blockIdx.x*blockDim.x + threadIdx.x;
        write_data[global_y*gridDim.x*blockDim.x + global_x] = read_data[threadIdx.x + TILE_SIZE*threadIdx.y];
    __syncthreads();
}
__device__ void load_full(float* read_data,float* write_data)
{
    int global_y = blockIdx.y*blockDim.y + threadIdx.y;
    int global_x = blockIdx.x*blockDim.x + threadIdx.x;
        write_data[threadIdx.x + TILE_SIZE*threadIdx.y] = read_data[global_y*gridDim.x*blockDim.x + global_x];
    __syncthreads();
}
__device__ void store_lower(float* read_data,float* write_data)
{
    int global_y = blockIdx.y*blockDim.y + threadIdx.y;
    int global_x = blockIdx.x*blockDim.x + threadIdx.x;
    if(threadIdx.y >= threadIdx.x)
        write_data[global_y*gridDim.x*blockDim.x + global_x] = read_data[threadIdx.x + TILE_SIZE*threadIdx.y];
    else
        write_data[global_y*gridDim.x*blockDim.x + global_x] = 0.0;
    __syncthreads();
}
__device__ void load_lower(float* read_data,float* write_data)
{
    int global_y = blockIdx.y*blockDim.y + threadIdx.y;
    int global_x = blockIdx.x*blockDim.x + threadIdx.x;
    if(threadIdx.y >= threadIdx.x)
        write_data[threadIdx.x + TILE_SIZE*threadIdx.y] = read_data[global_y*gridDim.x*blockDim.x + global_x];
    else
        write_data[threadIdx.x + TILE_SIZE*threadIdx.y] = 0.0;
    __syncthreads();
}
__device__ void potrf_tile(float* t_A)
{
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    for(int k=0;k<TILE_SIZE;k++)
    {
        if(t_x==t_y && t_x==k)
            t_A[k*TILE_SIZE + k] = sqrtf(t_A[k*TILE_SIZE + k]);
        __syncthreads();
        if(t_x<t_y && t_x == k && t_x<TILE_SIZE && t_y<TILE_SIZE)
        {
            t_A[t_y*TILE_SIZE + k]/= t_A[k*TILE_SIZE + k];
        }
        __syncthreads();
        if(t_x<t_y && t_x>k && t_x<TILE_SIZE && t_y<TILE_SIZE)
        {
            t_A[t_y*TILE_SIZE + t_x]-= t_A[t_x*TILE_SIZE + k]*t_A[t_y*TILE_SIZE + k];
        }
        __syncthreads();
    }
}
__device__ void trsm_tile(float *t_A1,float *t_A2)
{
	int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    __shared__ float temp[TILE_SIZE];                        // Using shared memory to Optimize
	for(int i=0;i<TILE_SIZE;i++)
    {
		if(t_x==i)
        {
			t_A2[t_y*TILE_SIZE + i]/= t_A1[i];
            temp[t_y] = t_A2[t_y*TILE_SIZE + i];
		}
		__syncthreads();
		if(t_x > i)
        {
			t_A2[t_y*TILE_SIZE+t_x]-= temp[t_x]*temp[t_y];
		}
		__syncthreads();
	}
}
__device__ void syrk_tile(float* rA1,float* rA2) 
{
    int t_y = threadIdx.y;
    int t_x = threadIdx.x;
    __shared__ float temp1[TILE_SIZE][TILE_SIZE+1];                        // Using shared memory to Optimize
    temp1[t_y][t_x] = rA1[t_y*TILE_SIZE+t_x];
    __syncthreads();
    float valueToSubtract = 0.0;
    for(int k=0;k<TILE_SIZE;k++)
    {
        valueToSubtract+= temp1[t_x][k]*temp1[t_y][k];
    }
    rA2[t_y*TILE_SIZE + t_x]-= valueToSubtract;
}
__global__ void right_looking_kernel(float* read_data,int N)
{
    __device__ __shared__ float temp2[TILE_SIZE*TILE_SIZE];
    __device__ __shared__ float temp3[TILE_SIZE];
    __device__ __shared__ float temp4[TILE_SIZE*TILE_SIZE];
    __device__ __shared__ int i;
    __device__ __shared__ int j;
    __device__ __shared__ int t1;
    t1 = -1;
    for(i=0;i<N/TILE_SIZE;i++)
    {
        if(blockIdx.x==i && blockIdx.y==i)
        {
            load_lower(read_data,temp2);
            potrf_tile(temp2);
            store_lower(temp2,read_data);
            for(j=0;j<TILE_SIZE;j++)
                temp3[j] = temp2[j*TILE_SIZE + j];
            t1=i;
        }
        __syncthreads();
        if(blockIdx.x==t1 && blockIdx.y>t1)
        {
            load_full(read_data,temp2);
            trsm_tile(temp3,temp2);
            store_full(temp2,read_data);
        }
        __syncthreads();
        if(blockIdx.x>t1 && blockIdx.y>t1)
        {
            load_full(read_data,temp4);
            syrk_tile(temp2,temp4);
            store_full(temp4,read_data);
        }          
    }
}
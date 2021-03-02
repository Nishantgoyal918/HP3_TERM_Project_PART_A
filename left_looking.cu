#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 32 // NB // Block SIZE

/*
 * Function to perform rank-k update 
 * half of the threads working
 */
__device__ void ssyrk_tile(float* rA1, float* rA2) 
{
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int column = blockIdx.x * TILE_SIZE + threadIdx.x;

    if(column <= row)
    {
        float updatedValue = rA2[row * TILE_SIZE + column];

        for(int k=0; i<TILE_SIZE; k++)
        {
            updatedValue -= rA1[row * TILE_SIZE + k] * rA1[column * TILE_SIZE + k];
        }

        rA2[row * TILE_SIZE + column] = updatedValue;
    }
}


/*
 * Function to perform general matrix multiplication 
 * DOUBT: I think calculation is given wrong in paper it should be rA2[k][n] 
 */
__device__ void sgemm_tile(const float* rA1, const float* rA2, float* rA3)
{
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int column = blockIdx.x * TILE_SIZE + threadIdx.x;


    float updatedValue = rA3[row * TILE_SIZE + column];

    for(int i=0; i<TILE_SIZE; i++)
    {
        updatedValue -= rA1[row * TILE_SIZE + i] * rA2[i * TILE_SIZE + column];
    }

    rA3[row * TILE_SIZE + column] = updatedValue;
}


/*
 * Function to store full tile from shared memory to global memory  
 */
__device__ void store_full(const float* s_data, float* g_data)
{
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int column = blockIdx.x * TILE_SIZE + threadIdx.x;

    g_data[row * TILE_SIZE + column] = s_data[row * TILE_SIZE + column];

    __syncthreads();
}


/*
 * Function to store lower triangular tile from shared memory to global memory  
 */
__device__ void store_lower(const float* s_data, float* g_data)
{
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int column = blockIdx.x * TILE_SIZE + threadIdx.x;

    if(column <= row)
        g_data[row * TILE_SIZE + column] = s_data[row * TILE_SIZE + column];
    else
        g_data[row * TILE_SIZE + column] = 0;

    __syncthreads();
}
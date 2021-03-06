#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>       // needed for the function sqrtf()

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
     int g_row = blockIdx.y * TILE_SIZE + threadIdx.y;
     int g_column = blockIdx.x * TILE_SIZE + threadIdx.x;
 
     int l_row = threadIdx.y;
     int l_column = threadIdx.x;
 
     g_data[g_row * TILE_SIZE + g_column] = s_data[l_row * TILE_SIZE + l_column];
 
     __syncthreads();
 }


/*
 * Function to store lower triangular tile from shared memory to global memory  
 */
 __device__ void store_lower(const float* s_data, float* g_data)
 {
     int g_row = blockIdx.y * TILE_SIZE + threadIdx.y;
     int g_column = blockIdx.x * TILE_SIZE + threadIdx.x;
 
     int l_row = threadIdx.y;
     int l_column = threadIdx.x;
 
     if(column <= row)
         g_data[g_row * TILE_SIZE + g_column] = s_data[l_row * TILE_SIZE + l_column];
     else
         g_data[g_row * TILE_SIZE + g_column] = 0;
 
     __syncthreads();
 }



/*
 * Function to perform Choleshky Factorization for a tile
 */
 __device__ void spotrf_tile(float* t_A)
 {
     int ty = blockIdx.x*blockDim.x + threadIdx.x;  // col
     int tx = blockIdx.y*blockDim.y + threadIdx.y; // row
 
     for(int k{0};k<TILE_SIZE;k++){
         // square root of diagonal elements
 
         if(tx==0 && ty==0)
             t_A[k*TILE_SIZE + k] = sqrtf(t_A[k*TILE_SIZE + k]);
         __syncthreads();
 
         // division step done parallaly
         if(ty<=tx && tx<TILE_SIZE - 1 && ty<TILE_SIZE - 1 && ty == k)
         {
             t_A[(tx+1)*TILE_SIZE + k]/= t_A[k*TILE_SIZE + k];
         }
         __syncthreads();
 
         if(ty<=tx && tx<TILE_SIZE - 1 && ty<TILE_SIZE - 1 && ty >= k)
         {
             t_A[(tx+1)*TILE_SIZE + (ty+1)]-= t_A[(tx+1)*TILE_SIZE + k]*t_A[(ty+1)*TILE_SIZE + k];
         }
         __syncthreads();
     }
 }

/*
* Function to perform triangular solve for a tile 
*/

__device__ void strsm_tile(float *t_A1, float *t_A2)
{
	// t_A2 is current unkonown 
	int ty = blockIdx.x*TILE_SIZE + threadIdx.x;
	int tx = blockIdx.y*TILE_SIZE + threadIdx.y;
	
	for(int i{0};i<TILE_SIZE;i++){
		if(ty==0){
			t_A2[tx*TILE_SIZE + i]/= t_A1[i*TILE_SIZE + i];
		}
		__syncthreads();

		if(ty>i && i<TILE_SIZE-1)
        {
			t_A2[tx*TILE_SIZE+ty]-= t_A2[tx*TILE_SIZE + i]*t_A1[ty*TILE_SIZE + i];
		}
		__syncthreads();
	}
 
}

/*
* Function to load a full tile from global memory to shared memory
*/

__device__ void load_full(float *t_A,float * S_A)
{
    // assigning a 2-D array in shared memory 

    int g_ty = blockIdx.x*blockDim.x + threadIdx.x;  // col
    int g_tx = blockIdx.y*blockDim.y + threadIdx.y; // row

    int l_tx = threadIdx.x;
    int l_ty = threadIdx.y;

    if(tx<TILE_SIZE && ty<TILE_SIZE)
        S_A[l_tx * TILE_SIZE + l_ty] = t_A[tx*TILE_SIZE + ty];
    __syncthreads();

}

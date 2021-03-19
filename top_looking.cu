/*Top Looking tiled implementation of Cholesky factorization*/

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>       

#define TILE_SIZE 32    // Tile Size(nb)

/*The algorithm and code given in the main reference paper have been followed*/
/*All matrices stored and accessed in row major form*/

/*Function to perform rank-k update */
__device__ void ssyrk_tile(float* rA1, float* rA2) 
{
    int row = threadIdx.y;
    int column = threadIdx.x;

    if(column <= row)
    {
        float updatedValue = rA2[row * TILE_SIZE + column];

        for(int k=0; k<TILE_SIZE; k++)
        {
            updatedValue -= rA1[row * TILE_SIZE + k] * rA1[column * TILE_SIZE + k];
        }

        rA2[row * TILE_SIZE + column] = updatedValue;
    }
}

/*General Matrix Multiplication*/
__device__ void sgemm_tile(float* rA1, float* rA2, float* rA3)
{
    int row = threadIdx.y;
    int column = threadIdx.x;    


    float updatedValue = rA3[row * TILE_SIZE + column];

    for(int i=0; i<TILE_SIZE; i++)
    {
        updatedValue -= rA1[row * TILE_SIZE + i] * rA2[column*TILE_SIZE + i];
    }

    rA3[row * TILE_SIZE + column] = updatedValue;
}

/*Function to perform Cholesky Factorization for a tile*/
 __device__ void spotrf_tile(float* t_A)
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


/*Function to perform triangular solve for a tile */

__device__ void strsm_tile(float *t_A1, float *t_A2)
{
	int ty = threadIdx.x;
	int tx = threadIdx.y;
	
	for(int i{0};i<TILE_SIZE;i++)
  {
		if(ty==0)
    {
			t_A2[tx*TILE_SIZE + i] /= t_A1[i*TILE_SIZE + i];
		}
		__syncthreads();

		if(ty>i && ty<TILE_SIZE-1)
    {
			t_A2[tx*TILE_SIZE+ty] -= (t_A2[tx*TILE_SIZE + i]*t_A1[ty*TILE_SIZE + i]);
		}
		__syncthreads();
	}
 
}

__global__ void load_full_tile(int m, int n, float* g_in, float* arr, int N)
{
    int  i = m*TILE_SIZE + threadIdx.y;
    int  j = n*TILE_SIZE + threadIdx.x;
    arr[threadIdx.y*TILE_SIZE + threadIdx.x] = g_in[i*N + j];
    __syncthreads(); 
}

__device__ void store_full_tile(int m, int n, float* g_in, float* arr, int N)
{
    int  i = m*TILE_SIZE + threadIdx.y;
    int  j = n*TILE_SIZE + threadIdx.x;
    g_in[i*N + j] = arr[threadIdx.y*TILE_SIZE + threadIdx.x];
    __syncthreads(); 
}

__global__ void load_lower_tile(int m, int n, float* g_in, float* arr, int N)
{
    int  i = m*TILE_SIZE + threadIdx.y;
    int  j = n*TILE_SIZE + threadIdx.x;
    if(threadIdx.x<=threadIdx.y)
     arr[threadIdx.y*TILE_SIZE + threadIdx.x] = g_in[i*N + j];
    else arr[threadIdx.y*TILE_SIZE + threadIdx.x] = 0.0f;
    __syncthreads(); 
}

__device__ void store_lower_tile(int m, int n, float* g_in, float* arr, int N)
{
    int  i = m*TILE_SIZE + threadIdx.y;
    int  j = n*TILE_SIZE + threadIdx.x;
    if(threadIdx.x<=threadIdx.y)
     g_in[i*N + j] = arr[threadIdx.y*TILE_SIZE + threadIdx.x];
    else g_in[i*N + j] = 0.0f;
    __syncthreads(); 
}

/*Function to perform some of the operations (in parallel) involved in updating the stripe of matrix to the left of current diagonal block*/

__global__ void step1_ops(float* g_in, int kk, int nn, int N, float* rA3)
 {
     __shared__ float rA1[TILE_SIZE*TILE_SIZE], rA2[TILE_SIZE*TILE_SIZE];
     int bx = blockIdx.x, i,j;
     i = kk*TILE_SIZE + threadIdx.y;
     j = bx*TILE_SIZE + threadIdx.x;
     rA1[threadIdx.y*TILE_SIZE + threadIdx.x] = g_in[i*N + j];
     __syncthreads(); 
          
     i = nn*TILE_SIZE + threadIdx.y;
     j = bx*TILE_SIZE + threadIdx.x;
     rA2[threadIdx.y*TILE_SIZE + threadIdx.x] = g_in[i*N + j];
     __syncthreads();
    
     sgemm_tile(rA1, rA2, rA3); 
     __syncthreads();
 }   

/*Function which completes updating the stripe of matrix after step1_ops() is done*/

__global__ void update_block(float* rA3, int kk, float* g_in, int N, int nn)
{
    __shared__ float rA1[TILE_SIZE*TILE_SIZE];
   
    int i,j;
    i = nn*TILE_SIZE + threadIdx.y;
    j = nn*TILE_SIZE + threadIdx.x;
    if(threadIdx.x<=threadIdx.y)
     rA1[threadIdx.y*TILE_SIZE + threadIdx.x] = g_in[i*N + j];
    else rA1[threadIdx.y*TILE_SIZE + threadIdx.x] = 0.0f;
    __syncthreads();
 
    strsm_tile(rA1,rA3);
    __syncthreads();
 
    store_full_tile(kk,nn,g_in,rA3,N);
}

/*Function to update the current diagonal triangle using the updated stripe of matrix (in parallel) obtained after update_block()*/

__global__ void step2_ops(float* g_in, int kk, float* rA1, int N)
{
    int bx = blockIdx.x;   
    __shared__ float rA2[TILE_SIZE*TILE_SIZE];
    int i = kk*TILE_SIZE + threadIdx.y;
    int j = bx*TILE_SIZE + threadIdx.x;
    rA2[threadIdx.y*TILE_SIZE + threadIdx.x] = g_in[i*N + j];
    __syncthreads();
 
    ssyrk_tile(rA2,rA1);
    __syncthreads();
}

/*Function to factor the diagonal triangle and update the matrix*/

__global__ void factor_triangle(float* g_in, int kk, float* rA1, int N)
{
    spotrf_tile(rA1);
    __syncthreads();
    store_lower_tile(kk,kk,g_in,rA1,N);
}

/*Function to set the entries of upper triangular block of the whole matrix to zero, after lower triangular block has been computed by Cholesky factorization*/ 

__global__ void set_to_zero(float* g_in, int N)
{
    //int bx = blockIdx.x + kk + 1;
    int bx,by;
    bx = blockIdx.x;  by = blockIdx.y;
    int i = by*TILE_SIZE + threadIdx.y;
    int j = bx*TILE_SIZE + threadIdx.x;
    if(bx > by)
     g_in[i*N+j] = 0.0f;
    __syncthreads();
}

void print_matrix(float *A,int m,int n)
{
    for(int i =0;i<m;i++)
    {
        for(int j=0;j<n;j++)
            printf("%.4f ",A[i*n+j]);
        printf("\n");
    }
}

/*Function to combine all the operations of Cholesky factorization*/

void launch_kernel(float* h_mat, int N)
{
    size_t size = N*N*sizeof(float);
    float* d_mat;
    cudaMalloc((void **)&d_mat, size); 
    cudaMemcpy(d_mat, h_mat, size, cudaMemcpyHostToDevice);   
    
    float *rA1, *rA3;
    cudaMalloc((void **)&rA1, TILE_SIZE*TILE_SIZE*sizeof(float));
    cudaMalloc((void **)&rA3, TILE_SIZE*TILE_SIZE*sizeof(float));
    int nn,kk;
 
    for(kk=0; kk<N/TILE_SIZE; kk++)
    {
      dim3 grid1(1,1,1);
      dim3 block(TILE_SIZE, TILE_SIZE, 1);
      
      for(nn=0; nn<kk; nn++)
      {
        load_full_tile<<<grid1, block>>> (kk,nn,d_mat,rA3,N);   
        dim3 grid(nn, 1, 1);                  
        
        if(nn!=0)
         step1_ops<<<grid, block>>>(d_mat,kk,nn, N, rA3); 
        
        update_block<<<grid1,block>>> (rA3,kk,d_mat,N,nn);
      } 
      
      dim3 grid(kk,1,1);
      load_lower_tile<<<grid1,block>>> (kk,kk,d_mat,rA1,N);
      
      if(kk>0)
       step2_ops<<<grid,block>>> (d_mat,kk,rA1,N); 
     
      factor_triangle<<<grid1,block>>> (d_mat,kk,rA1,N);
    }
    
    dim3 grid(N/TILE_SIZE, N/TILE_SIZE, 1);
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    set_to_zero<<<grid,block>>> (d_mat,N);
     
    cudaMemcpy(h_mat, d_mat, size, cudaMemcpyDeviceToHost);
}

int main()
 {
    int N;
    printf("Enter order of matrix: ");
    scanf("%d", &N);
    size_t size = N*N*sizeof(float);
    float* h_mat = (float*) malloc(size);
    
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            scanf("%f",&h_mat[i*N + j]);
        }
    }
    print_matrix(h_mat, N, N);
        
    printf("\nPerforming top looking cholesky factorization...\n\n");
    
    launch_kernel(h_mat, N);
    print_matrix(h_mat, N, N);
    return  0;
}

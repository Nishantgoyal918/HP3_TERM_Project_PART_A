#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<cmath>
#define TILE_SIZE 32            //Tile size and block size, both are taken as 32
__global__ void store_full(float*,float*,int,int,int);
__global__ void load_full(float*,float*,int,int,int);
__global__ void store_lower(float*,float*,int,int,int);
__global__ void load_lower(float*,float*,int,int,int);
__global__ void potrf_tile(float*);
__global__ void trsm_tile(float*,int,int,int);
__global__ void syrk_tile(float*,float*,int,int,int);
void right_looking_launch_kernel(float*,int);

__global__ void store_full(float* read_data,float* write_data,int i,int j,int N)
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = i*blockDim.x + threadIdx.x;
        write_data[global_y*N + global_x] = read_data[threadIdx.x + TILE_SIZE*threadIdx.y];
    __syncthreads();
}
__global__ void load_full(float* read_data,float* write_data,int i,int j,int N)
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = i*blockDim.x + threadIdx.x;
        write_data[threadIdx.x + TILE_SIZE*threadIdx.y] = read_data[global_y*N + global_x];
    __syncthreads();
}
__global__ void store_lower(float* read_data,float* write_data,int i,int j,int N)
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = i*blockDim.x + threadIdx.x;
    if(threadIdx.y >= threadIdx.x)
        write_data[global_y*N + global_x] = read_data[threadIdx.x + TILE_SIZE*threadIdx.y];
    else
        write_data[global_y*N + global_x] = 0.0;
    __syncthreads();
}
__global__ void load_lower(float* read_data,float* write_data,int i,int j,int N)
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = i*blockDim.x + threadIdx.x;
    if(threadIdx.y >= threadIdx.x)
        write_data[threadIdx.x + TILE_SIZE*threadIdx.y] = read_data[global_y*N + global_x];
    else
        write_data[threadIdx.x + TILE_SIZE*threadIdx.y] = 0.0;
    __syncthreads();
}
__global__ void potrf_tile(float* t_A)
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
__global__ void trsm_tile(float *read_data,int i,int j,int N)
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
__global__ void syrk_tile(float* read_data,float* rA2,int i,int j,int k,int N) 
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = k*blockDim.x + threadIdx.x;
    int t_y = threadIdx.y;
    int t_x = threadIdx.x;
    /*
    __shared__ float temp0[TILE_SIZE][TILE_SIZE+1];                        // Using shared memory to Optimize
    __shared__ float temp1[TILE_SIZE][TILE_SIZE+1];                        // Using shared memory to Optimize
    temp0[t_y][t_x] = read_data[global_x*N + i*blockDim.x + t_y];
    temp1[t_x][t_y] = read_data[global_y*N + i*blockDim.x + t_x];
    __syncthreads();
    */
    float valueToSubtract = 0.0;
    for(int r=0;r<TILE_SIZE;r++)
    {
        valueToSubtract+= read_data[global_x*N + i*blockDim.x + r]*read_data[global_y*N + i*blockDim.x + r];//temp0[r][t_x]*temp1[r][t_y];
    }
    rA2[t_y*TILE_SIZE + t_x]-= valueToSubtract;
    __syncthreads();
}
void right_looking_launch_kernel(float* M,int N)
{
    cudaError_t err = cudaSuccess;
    float *read_data = NULL;
    err = cudaMalloc((void **)&read_data,N*N*sizeof(float));
    if(err != cudaSuccess)
    {
        fprintf(stderr,"Failed to allocate matrix M on the CUDA device! (error code %s)\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Coping the matrix M from host memory to device memory\n");
    err = cudaMemcpy(read_data,M,N*N*sizeof(float),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        fprintf(stderr,"Failed to copy matrix M from host to device (error code %s)\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float* print_block_data = (float*)malloc(TILE_SIZE*TILE_SIZE*sizeof(float));
    if(print_block_data == NULL)
    {
        fprintf(stderr,"Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    float* print_read_data = (float*)malloc(N*N*sizeof(float));
    if(print_read_data == NULL)
    {
        fprintf(stderr,"Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    float* block_data = NULL;
    err = cudaMalloc((void **)&block_data,TILE_SIZE*TILE_SIZE*sizeof(float));
    if(err != cudaSuccess)
    {
        fprintf(stderr,"Failed to allocate matrix M on the CUDA device! (error code %s)\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    int i,j,k;
    dim3 grid(1,1,1);
    dim3 block(TILE_SIZE,TILE_SIZE,1);
    for(i=0;i<N/TILE_SIZE;i++)
    {
        load_lower<<<grid,block>>>(read_data,block_data,i,i,N);
        err = cudaGetLastError();
        if(err != cudaSuccess)
        {
            fprintf(stderr,"Failed to launch CUDA kernel (error code %s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        /*
        err = cudaMemcpy(print_block_data,block_data,TILE_SIZE*TILE_SIZE*sizeof(float),cudaMemcpyDeviceToHost);
        if(err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy the output matrix M from device to Host (error code %s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        for(int r=0;r<TILE_SIZE;r++)
        {
            for(int s=0;s<TILE_SIZE;s++)
                printf("%f\t",print_block_data[s + r*TILE_SIZE]);
            printf("\n");
        }
        printf("%d\n",i);
        */
        potrf_tile<<<grid,block>>>(block_data);
        err = cudaGetLastError();
        if(err != cudaSuccess)
        {
            fprintf(stderr,"Failed to launch CUDA kernel (error code %s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        /*
        err = cudaMemcpy(print_block_data,block_data,TILE_SIZE*TILE_SIZE*sizeof(float),cudaMemcpyDeviceToHost);
        if(err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy the output matrix M from device to Host (error code %s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        for(int r=0;r<TILE_SIZE;r++)
        {
            for(int s=0;s<TILE_SIZE;s++)
                printf("%f\t",print_block_data[s + r*TILE_SIZE]);
            printf("\n");
        }
        printf("%d\n",i);
        */
        store_lower<<<grid,block>>>(block_data,read_data,i,i,N);
        err = cudaGetLastError();
        if(err != cudaSuccess)
        {
            fprintf(stderr,"Failed to launch CUDA kernel (error code %s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        /*
        err = cudaMemcpy(print_read_data,read_data,N*N*sizeof(float),cudaMemcpyDeviceToHost);
        if(err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy the output matrix M from device to Host (error code %s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        for(int r=0;r<N;r++)
        {
            for(int s=0;s<N;s++)
                printf("%f\t",print_read_data[s + r*N]);
            printf("\n");
        }
        printf("%d\n",i);
        */
        for(j=i+1;j<N/TILE_SIZE;j++)
        {
            trsm_tile<<<grid,block>>>(read_data,i,j,N);
            err = cudaGetLastError();
            if(err != cudaSuccess)
            {
                fprintf(stderr,"Failed to launch CUDA kernel (error code %s)\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            /*
            err = cudaMemcpy(print_block_data,block_data,TILE_SIZE*TILE_SIZE*sizeof(float),cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy the output matrix M from device to Host (error code %s)\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            for(int r=0;r<TILE_SIZE;r++)
            {
                for(int s=0;s<TILE_SIZE;s++)
                    printf("%f\t",print_block_data[s + r*TILE_SIZE]);
                printf("\n");
            }
            printf("%d\n",i);
            */
            for(k=i+1;k<((N/TILE_SIZE)-1);k++)
            {
                load_full<<<grid,block>>>(read_data,block_data,k,j,N);
                err = cudaGetLastError();
                if(err != cudaSuccess)
                {
                    fprintf(stderr,"Failed to launch CUDA kernel (error code %s)\n", cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }
                syrk_tile<<<grid,block>>>(read_data,block_data,i,j,k,N);
                err = cudaGetLastError();
                if(err != cudaSuccess)
                {
                    fprintf(stderr,"Failed to launch CUDA kernel (error code %s)\n", cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }
                store_full<<<grid,block>>>(block_data,read_data,k,j,N);
                err = cudaGetLastError();
                if(err != cudaSuccess)
                {
                    fprintf(stderr,"Failed to launch CUDA kernel (error code %s)\n", cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }
            }
            load_lower<<<grid,block>>>(read_data,block_data,k,j,N);
            err = cudaGetLastError();
            if(err != cudaSuccess)
            {
                fprintf(stderr,"Failed to launch CUDA kernel (error code %s)\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }            
            syrk_tile<<<grid,block>>>(read_data,block_data,i,j,k,N);
            err = cudaGetLastError();
            if(err != cudaSuccess)
            {
                fprintf(stderr,"Failed to launch CUDA kernel (error code %s)\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            store_lower<<<grid,block>>>(block_data,read_data,k,j,N);
            err = cudaGetLastError();
            if(err != cudaSuccess)
            {
                fprintf(stderr,"Failed to launch CUDA kernel (error code %s)\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
        }
    }
    err = cudaMemcpy(M,read_data,N*N*sizeof(float),cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy the output matrix M from device to Host (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(block_data);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix M (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(read_data);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix M (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceReset();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the CUDA device (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    free(print_block_data);
    free(print_read_data);
}
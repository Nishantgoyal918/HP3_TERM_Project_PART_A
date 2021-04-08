#include "right_looking.cu"
int main()
{
    int D,N;
    printf("Enter number of matrix (D) : ");
    scanf("%d",&D);
    printf("Enter dimension (N) : ");
    scanf("%d",&N);
    size_t size = D*N*N*sizeof(float);
    float *M = (float *)malloc(size);
    if(M == NULL)
    {
        fprintf(stderr,"Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    int i,j,m;
    printf("Enter input matrix: \n");
    for(m=0;m<D;m++)
    {
        for(i=0;i<N;i++)
        {
            for(j=0;j<N;j++)
            {
                scanf("%f",&M[i*N+j+m*N*N]);
            }
        }
    }
    cudaError_t err = cudaSuccess;
    float *read_data = NULL;
    err = cudaMalloc((void **)&read_data,size);
    if(err != cudaSuccess)
    {
        fprintf(stderr,"Failed to allocate matrix on the CUDA device! (error code %s)\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Coping the matrix from host memory to device memory\n");
    err = cudaMemcpy(read_data,M,size,cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        fprintf(stderr,"Failed to copy matrix from host to device (error code %s)\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    dim3 grid(D,1,1);
    dim3 block(TILE_SIZE,TILE_SIZE,1);
    size_t shared_size = (N*N)*sizeof(float);
    right_looking_launch_kernel<<<grid,block,shared_size>>>(read_data,N);
    err = cudaMemcpy(M,read_data,size,cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy the output matrix M from device to Host (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Printing output matrix\n");
    for(m=0;m<D;m++)
    {
        for(i=0;i<N;i++)
        {
            for(j=0;j<N;j++)
            {
                
                if(j>i)
                    M[i*N+j+m*N*N] = 0;
                printf("%f ",M[i*N+j+m*N*N]);
            }
            printf("\n");
        }
        printf("\n");
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
    free(M);
    printf("DONE!\n");
    return 0;
}
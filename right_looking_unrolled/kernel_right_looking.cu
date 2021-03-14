#include "right_looking.cu"
int main()
{
    cudaError_t err = cudaSuccess;
    int N;
    printf("Enter dimension (N) : ");
    scanf("%d",&N);
    printf("Testing for matrix M [%dx%d]\n",N,N);
    size_t size = N*N*sizeof(float);
    float *M = (float *)malloc(size);
    if(M == NULL)
    {
        fprintf(stderr,"Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    int i,j;
    printf("Enter input matrix: \n");
    for(i=0;i<N;i++)
    {
        for(j=0;j<N;j++)
        {
            scanf("%f",&M[i*N + j]);
        }
    }
    float *d_M = NULL;
    err = cudaMalloc((void **)&d_M,size);
    if(err != cudaSuccess)
    {
        fprintf(stderr,"Failed to allocate matrix M on the CUDA device! (error code %s)\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Coping the matrix M from host memory to device memory\n");
    err = cudaMemcpy(d_M,M,size,cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        fprintf(stderr,"Failed to copy matrix M from host to device (error code %s)\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    dim3 grid(N/TILE_SIZE,N/TILE_SIZE,1);
    dim3 block(TILE_SIZE,TILE_SIZE,1);
    right_looking_kernel<<<grid,block>>>(d_M,N);//,temp2,temp3,temp4);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        fprintf(stderr,"Failed to launch CUDA kernel (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(M,d_M,size,cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy the output matrix M from device to Host (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Printing output matrix\n");
    for(i=0;i<N;i++)
    {
        for(j=0;j<N;j++)
        {
            if(j<=i)
                printf("%f\t",M[i*N + j]);
            else
                printf("%f\t",0.0);
        }
        printf("\n");
    }
    err = cudaFree(d_M);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix M (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    free(M);
    err = cudaDeviceReset();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the CUDA device (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("DONE!\n");
    return 0;
}
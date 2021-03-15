#include "right_looking.cu"
int main()
{
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
    right_looking_launch_kernel(M,N);
    
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
    
    free(M);
    printf("DONE!\n");
    return 0;
}
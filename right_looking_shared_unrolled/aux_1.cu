// TILE_SIZE and N are variable/parameter here
#define TILE_SIZE 4

__device__ void store_full_row(float* read_data,float* write_data,int i,int N)
{
    int global_y;
    int global_x = i*blockDim.x + threadIdx.x;
        
    global_y = 0*blockDim.y + threadIdx.y;
    write_data[global_y*N + global_x] = read_data[threadIdx.x + (TILE_SIZE+1)*global_y];
        
    global_y = 1*blockDim.y + threadIdx.y;
    write_data[global_y*N + global_x] = read_data[threadIdx.x + (TILE_SIZE+1)*global_y];
        
    global_y = 2*blockDim.y + threadIdx.y;
    write_data[global_y*N + global_x] = read_data[threadIdx.x + (TILE_SIZE+1)*global_y];
        
    global_y = 3*blockDim.y + threadIdx.y;
    write_data[global_y*N + global_x] = read_data[threadIdx.x + (TILE_SIZE+1)*global_y];
    
    __syncthreads();
}
__device__ void load_full_row(float* read_data,float* write_data,int i,int N)
{
    int global_y;
    int global_x = i*blockDim.x + threadIdx.x;
        
    global_y = 0*blockDim.y + threadIdx.y;
    write_data[threadIdx.x + (TILE_SIZE+1)*global_y] = read_data[global_y*N + global_x];
        
    global_y = 1*blockDim.y + threadIdx.y;
    write_data[threadIdx.x + (TILE_SIZE+1)*global_y] = read_data[global_y*N + global_x];
        
    global_y = 2*blockDim.y + threadIdx.y;
    write_data[threadIdx.x + (TILE_SIZE+1)*global_y] = read_data[global_y*N + global_x];
        
    global_y = 3*blockDim.y + threadIdx.y;
    write_data[threadIdx.x + (TILE_SIZE+1)*global_y] = read_data[global_y*N + global_x];
    
    __syncthreads();
}
__device__ void potrf_tile(float* t_A)
{
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    __shared__ float temp2;
        
    if(t_x==t_y && t_x==0)
    {
        t_A[0*(TILE_SIZE+1) + 0] = sqrtf(t_A[0*(TILE_SIZE+1) + 0]);
        temp2 = t_A[0*(TILE_SIZE+1) + 0];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 0)
    {
        t_A[t_y*(TILE_SIZE+1) + 0]/= temp2;
    }
    __syncthreads();
    if(0<t_y && 0<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 0]*t_A[t_y*(TILE_SIZE+1) + 0];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==1)
    {
        t_A[1*(TILE_SIZE+1) + 1] = sqrtf(t_A[1*(TILE_SIZE+1) + 1]);
        temp2 = t_A[1*(TILE_SIZE+1) + 1];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 1)
    {
        t_A[t_y*(TILE_SIZE+1) + 1]/= temp2;
    }
    __syncthreads();
    if(1<t_y && 1<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 1]*t_A[t_y*(TILE_SIZE+1) + 1];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==2)
    {
        t_A[2*(TILE_SIZE+1) + 2] = sqrtf(t_A[2*(TILE_SIZE+1) + 2]);
        temp2 = t_A[2*(TILE_SIZE+1) + 2];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 2)
    {
        t_A[t_y*(TILE_SIZE+1) + 2]/= temp2;
    }
    __syncthreads();
    if(2<t_y && 2<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 2]*t_A[t_y*(TILE_SIZE+1) + 2];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==3)
    {
        t_A[3*(TILE_SIZE+1) + 3] = sqrtf(t_A[3*(TILE_SIZE+1) + 3]);
        temp2 = t_A[3*(TILE_SIZE+1) + 3];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 3)
    {
        t_A[t_y*(TILE_SIZE+1) + 3]/= temp2;
    }
    __syncthreads();
    if(3<t_y && 3<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 3]*t_A[t_y*(TILE_SIZE+1) + 3];
    }
    __syncthreads();
    
}
__device__ void trsm_tile(float *row_data,int i,int j,int N)
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = i*blockDim.x + threadIdx.x;
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
        
    if(t_x==0)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 0)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  0]*row_data[global_y*(TILE_SIZE+1) + 0];
    }
    __syncthreads();
        
    if(t_x==1)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 1)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  1]*row_data[global_y*(TILE_SIZE+1) + 1];
    }
    __syncthreads();
        
    if(t_x==2)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 2)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  2]*row_data[global_y*(TILE_SIZE+1) + 2];
    }
    __syncthreads();
        
    if(t_x==3)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 3)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  3]*row_data[global_y*(TILE_SIZE+1) + 3];
    }
    __syncthreads();
    
}
__device__ void syrk_tile(float* row_data,float* edit_data,int i,int j,int N) 
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = i*blockDim.x + threadIdx.x;
    int t_y = threadIdx.y;
    int t_x = threadIdx.x;
    float valueToSubtract = 0.0;
        
    valueToSubtract+= row_data[0 + global_y*(TILE_SIZE+1)]*row_data[0 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[1 + global_y*(TILE_SIZE+1)]*row_data[1 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[2 + global_y*(TILE_SIZE+1)]*row_data[2 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[3 + global_y*(TILE_SIZE+1)]*row_data[3 + global_x*(TILE_SIZE+1)];
        
    edit_data[t_y*(TILE_SIZE+1) + t_x]-= valueToSubtract;
    __syncthreads();
}
__device__ void store_zeros(float* A,int N)
{
    int t_y = threadIdx.y;
    int t_x = threadIdx.x;
                
    A[1*blockDim.x + t_x + (0*blockDim.y + t_y)*N] = 0.0;
            
    A[2*blockDim.x + t_x + (0*blockDim.y + t_y)*N] = 0.0;
            
    A[3*blockDim.x + t_x + (0*blockDim.y + t_y)*N] = 0.0;
                        
    A[2*blockDim.x + t_x + (1*blockDim.y + t_y)*N] = 0.0;
            
    A[3*blockDim.x + t_x + (1*blockDim.y + t_y)*N] = 0.0;
                        
    A[3*blockDim.x + t_x + (2*blockDim.y + t_y)*N] = 0.0;
            
    __syncthreads();
}

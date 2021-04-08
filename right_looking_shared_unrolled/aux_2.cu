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
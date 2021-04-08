__global__ void right_looking_launch_kernel(float* read_data,int N)
{
    __shared__ float tile_data[TILE_SIZE*TILE_SIZE];
    extern __shared__ float row_data[];
    int i,j,k;
    for(i=0;i<N/TILE_SIZE;i++)
    {
        load_lower(read_data,tile_data,i,i,N);
        potrf_tile(tile_data);
        store_lower(tile_data,read_data,i,i,N);
        load_full_row(read_data,row_data,i,N);
        for(j=i+1;j<N/TILE_SIZE;j++)
        {
            trsm_tile(row_data,i,j,N);
            for(k=i+1;k<j;k++)
            {
                load_full(read_data,tile_data,k,j,N);
                syrk_tile(row_data,tile_data,k,j,N);
                store_full(tile_data,read_data,k,j,N);
            }
            load_full(read_data,tile_data,k,j,N);
            syrk_tile(row_data,tile_data,k,j,N);
            store_full(tile_data,read_data,k,j,N);
        }
        store_full_row(row_data,read_data,i,N);
    }
    store_zeros(read_data,N);
}
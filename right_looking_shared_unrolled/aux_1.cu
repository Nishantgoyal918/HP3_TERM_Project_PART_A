// TILE_SIZE and N are variable/parameter here
#define TILE_SIZE 32

__device__ void store_full_row(float* read_data,float* write_data,int i,int N)
{
    int global_y;
    int global_x = i*blockDim.x + threadIdx.x;
        
    global_y = 0*blockDim.y + threadIdx.y;
    write_data[global_y*N + global_x] = read_data[threadIdx.x + (TILE_SIZE+1)*global_y];
    
    __syncthreads();
}
__device__ void load_full_row(float* read_data,float* write_data,int i,int N)
{
    int global_y;
    int global_x = i*blockDim.x + threadIdx.x;
        
    global_y = 0*blockDim.y + threadIdx.y;
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
        
    if(t_x==t_y && t_x==4)
    {
        t_A[4*(TILE_SIZE+1) + 4] = sqrtf(t_A[4*(TILE_SIZE+1) + 4]);
        temp2 = t_A[4*(TILE_SIZE+1) + 4];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 4)
    {
        t_A[t_y*(TILE_SIZE+1) + 4]/= temp2;
    }
    __syncthreads();
    if(4<t_y && 4<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 4]*t_A[t_y*(TILE_SIZE+1) + 4];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==5)
    {
        t_A[5*(TILE_SIZE+1) + 5] = sqrtf(t_A[5*(TILE_SIZE+1) + 5]);
        temp2 = t_A[5*(TILE_SIZE+1) + 5];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 5)
    {
        t_A[t_y*(TILE_SIZE+1) + 5]/= temp2;
    }
    __syncthreads();
    if(5<t_y && 5<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 5]*t_A[t_y*(TILE_SIZE+1) + 5];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==6)
    {
        t_A[6*(TILE_SIZE+1) + 6] = sqrtf(t_A[6*(TILE_SIZE+1) + 6]);
        temp2 = t_A[6*(TILE_SIZE+1) + 6];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 6)
    {
        t_A[t_y*(TILE_SIZE+1) + 6]/= temp2;
    }
    __syncthreads();
    if(6<t_y && 6<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 6]*t_A[t_y*(TILE_SIZE+1) + 6];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==7)
    {
        t_A[7*(TILE_SIZE+1) + 7] = sqrtf(t_A[7*(TILE_SIZE+1) + 7]);
        temp2 = t_A[7*(TILE_SIZE+1) + 7];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 7)
    {
        t_A[t_y*(TILE_SIZE+1) + 7]/= temp2;
    }
    __syncthreads();
    if(7<t_y && 7<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 7]*t_A[t_y*(TILE_SIZE+1) + 7];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==8)
    {
        t_A[8*(TILE_SIZE+1) + 8] = sqrtf(t_A[8*(TILE_SIZE+1) + 8]);
        temp2 = t_A[8*(TILE_SIZE+1) + 8];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 8)
    {
        t_A[t_y*(TILE_SIZE+1) + 8]/= temp2;
    }
    __syncthreads();
    if(8<t_y && 8<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 8]*t_A[t_y*(TILE_SIZE+1) + 8];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==9)
    {
        t_A[9*(TILE_SIZE+1) + 9] = sqrtf(t_A[9*(TILE_SIZE+1) + 9]);
        temp2 = t_A[9*(TILE_SIZE+1) + 9];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 9)
    {
        t_A[t_y*(TILE_SIZE+1) + 9]/= temp2;
    }
    __syncthreads();
    if(9<t_y && 9<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 9]*t_A[t_y*(TILE_SIZE+1) + 9];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==10)
    {
        t_A[10*(TILE_SIZE+1) + 10] = sqrtf(t_A[10*(TILE_SIZE+1) + 10]);
        temp2 = t_A[10*(TILE_SIZE+1) + 10];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 10)
    {
        t_A[t_y*(TILE_SIZE+1) + 10]/= temp2;
    }
    __syncthreads();
    if(10<t_y && 10<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 10]*t_A[t_y*(TILE_SIZE+1) + 10];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==11)
    {
        t_A[11*(TILE_SIZE+1) + 11] = sqrtf(t_A[11*(TILE_SIZE+1) + 11]);
        temp2 = t_A[11*(TILE_SIZE+1) + 11];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 11)
    {
        t_A[t_y*(TILE_SIZE+1) + 11]/= temp2;
    }
    __syncthreads();
    if(11<t_y && 11<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 11]*t_A[t_y*(TILE_SIZE+1) + 11];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==12)
    {
        t_A[12*(TILE_SIZE+1) + 12] = sqrtf(t_A[12*(TILE_SIZE+1) + 12]);
        temp2 = t_A[12*(TILE_SIZE+1) + 12];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 12)
    {
        t_A[t_y*(TILE_SIZE+1) + 12]/= temp2;
    }
    __syncthreads();
    if(12<t_y && 12<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 12]*t_A[t_y*(TILE_SIZE+1) + 12];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==13)
    {
        t_A[13*(TILE_SIZE+1) + 13] = sqrtf(t_A[13*(TILE_SIZE+1) + 13]);
        temp2 = t_A[13*(TILE_SIZE+1) + 13];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 13)
    {
        t_A[t_y*(TILE_SIZE+1) + 13]/= temp2;
    }
    __syncthreads();
    if(13<t_y && 13<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 13]*t_A[t_y*(TILE_SIZE+1) + 13];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==14)
    {
        t_A[14*(TILE_SIZE+1) + 14] = sqrtf(t_A[14*(TILE_SIZE+1) + 14]);
        temp2 = t_A[14*(TILE_SIZE+1) + 14];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 14)
    {
        t_A[t_y*(TILE_SIZE+1) + 14]/= temp2;
    }
    __syncthreads();
    if(14<t_y && 14<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 14]*t_A[t_y*(TILE_SIZE+1) + 14];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==15)
    {
        t_A[15*(TILE_SIZE+1) + 15] = sqrtf(t_A[15*(TILE_SIZE+1) + 15]);
        temp2 = t_A[15*(TILE_SIZE+1) + 15];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 15)
    {
        t_A[t_y*(TILE_SIZE+1) + 15]/= temp2;
    }
    __syncthreads();
    if(15<t_y && 15<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 15]*t_A[t_y*(TILE_SIZE+1) + 15];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==16)
    {
        t_A[16*(TILE_SIZE+1) + 16] = sqrtf(t_A[16*(TILE_SIZE+1) + 16]);
        temp2 = t_A[16*(TILE_SIZE+1) + 16];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 16)
    {
        t_A[t_y*(TILE_SIZE+1) + 16]/= temp2;
    }
    __syncthreads();
    if(16<t_y && 16<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 16]*t_A[t_y*(TILE_SIZE+1) + 16];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==17)
    {
        t_A[17*(TILE_SIZE+1) + 17] = sqrtf(t_A[17*(TILE_SIZE+1) + 17]);
        temp2 = t_A[17*(TILE_SIZE+1) + 17];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 17)
    {
        t_A[t_y*(TILE_SIZE+1) + 17]/= temp2;
    }
    __syncthreads();
    if(17<t_y && 17<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 17]*t_A[t_y*(TILE_SIZE+1) + 17];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==18)
    {
        t_A[18*(TILE_SIZE+1) + 18] = sqrtf(t_A[18*(TILE_SIZE+1) + 18]);
        temp2 = t_A[18*(TILE_SIZE+1) + 18];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 18)
    {
        t_A[t_y*(TILE_SIZE+1) + 18]/= temp2;
    }
    __syncthreads();
    if(18<t_y && 18<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 18]*t_A[t_y*(TILE_SIZE+1) + 18];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==19)
    {
        t_A[19*(TILE_SIZE+1) + 19] = sqrtf(t_A[19*(TILE_SIZE+1) + 19]);
        temp2 = t_A[19*(TILE_SIZE+1) + 19];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 19)
    {
        t_A[t_y*(TILE_SIZE+1) + 19]/= temp2;
    }
    __syncthreads();
    if(19<t_y && 19<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 19]*t_A[t_y*(TILE_SIZE+1) + 19];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==20)
    {
        t_A[20*(TILE_SIZE+1) + 20] = sqrtf(t_A[20*(TILE_SIZE+1) + 20]);
        temp2 = t_A[20*(TILE_SIZE+1) + 20];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 20)
    {
        t_A[t_y*(TILE_SIZE+1) + 20]/= temp2;
    }
    __syncthreads();
    if(20<t_y && 20<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 20]*t_A[t_y*(TILE_SIZE+1) + 20];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==21)
    {
        t_A[21*(TILE_SIZE+1) + 21] = sqrtf(t_A[21*(TILE_SIZE+1) + 21]);
        temp2 = t_A[21*(TILE_SIZE+1) + 21];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 21)
    {
        t_A[t_y*(TILE_SIZE+1) + 21]/= temp2;
    }
    __syncthreads();
    if(21<t_y && 21<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 21]*t_A[t_y*(TILE_SIZE+1) + 21];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==22)
    {
        t_A[22*(TILE_SIZE+1) + 22] = sqrtf(t_A[22*(TILE_SIZE+1) + 22]);
        temp2 = t_A[22*(TILE_SIZE+1) + 22];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 22)
    {
        t_A[t_y*(TILE_SIZE+1) + 22]/= temp2;
    }
    __syncthreads();
    if(22<t_y && 22<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 22]*t_A[t_y*(TILE_SIZE+1) + 22];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==23)
    {
        t_A[23*(TILE_SIZE+1) + 23] = sqrtf(t_A[23*(TILE_SIZE+1) + 23]);
        temp2 = t_A[23*(TILE_SIZE+1) + 23];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 23)
    {
        t_A[t_y*(TILE_SIZE+1) + 23]/= temp2;
    }
    __syncthreads();
    if(23<t_y && 23<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 23]*t_A[t_y*(TILE_SIZE+1) + 23];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==24)
    {
        t_A[24*(TILE_SIZE+1) + 24] = sqrtf(t_A[24*(TILE_SIZE+1) + 24]);
        temp2 = t_A[24*(TILE_SIZE+1) + 24];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 24)
    {
        t_A[t_y*(TILE_SIZE+1) + 24]/= temp2;
    }
    __syncthreads();
    if(24<t_y && 24<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 24]*t_A[t_y*(TILE_SIZE+1) + 24];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==25)
    {
        t_A[25*(TILE_SIZE+1) + 25] = sqrtf(t_A[25*(TILE_SIZE+1) + 25]);
        temp2 = t_A[25*(TILE_SIZE+1) + 25];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 25)
    {
        t_A[t_y*(TILE_SIZE+1) + 25]/= temp2;
    }
    __syncthreads();
    if(25<t_y && 25<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 25]*t_A[t_y*(TILE_SIZE+1) + 25];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==26)
    {
        t_A[26*(TILE_SIZE+1) + 26] = sqrtf(t_A[26*(TILE_SIZE+1) + 26]);
        temp2 = t_A[26*(TILE_SIZE+1) + 26];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 26)
    {
        t_A[t_y*(TILE_SIZE+1) + 26]/= temp2;
    }
    __syncthreads();
    if(26<t_y && 26<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 26]*t_A[t_y*(TILE_SIZE+1) + 26];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==27)
    {
        t_A[27*(TILE_SIZE+1) + 27] = sqrtf(t_A[27*(TILE_SIZE+1) + 27]);
        temp2 = t_A[27*(TILE_SIZE+1) + 27];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 27)
    {
        t_A[t_y*(TILE_SIZE+1) + 27]/= temp2;
    }
    __syncthreads();
    if(27<t_y && 27<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 27]*t_A[t_y*(TILE_SIZE+1) + 27];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==28)
    {
        t_A[28*(TILE_SIZE+1) + 28] = sqrtf(t_A[28*(TILE_SIZE+1) + 28]);
        temp2 = t_A[28*(TILE_SIZE+1) + 28];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 28)
    {
        t_A[t_y*(TILE_SIZE+1) + 28]/= temp2;
    }
    __syncthreads();
    if(28<t_y && 28<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 28]*t_A[t_y*(TILE_SIZE+1) + 28];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==29)
    {
        t_A[29*(TILE_SIZE+1) + 29] = sqrtf(t_A[29*(TILE_SIZE+1) + 29]);
        temp2 = t_A[29*(TILE_SIZE+1) + 29];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 29)
    {
        t_A[t_y*(TILE_SIZE+1) + 29]/= temp2;
    }
    __syncthreads();
    if(29<t_y && 29<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 29]*t_A[t_y*(TILE_SIZE+1) + 29];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==30)
    {
        t_A[30*(TILE_SIZE+1) + 30] = sqrtf(t_A[30*(TILE_SIZE+1) + 30]);
        temp2 = t_A[30*(TILE_SIZE+1) + 30];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 30)
    {
        t_A[t_y*(TILE_SIZE+1) + 30]/= temp2;
    }
    __syncthreads();
    if(30<t_y && 30<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 30]*t_A[t_y*(TILE_SIZE+1) + 30];
    }
    __syncthreads();
        
    if(t_x==t_y && t_x==31)
    {
        t_A[31*(TILE_SIZE+1) + 31] = sqrtf(t_A[31*(TILE_SIZE+1) + 31]);
        temp2 = t_A[31*(TILE_SIZE+1) + 31];
    }
    __syncthreads();
    if(t_x<t_y && t_x == 31)
    {
        t_A[t_y*(TILE_SIZE+1) + 31]/= temp2;
    }
    __syncthreads();
    if(31<t_y && 31<t_x && t_x<=t_y)
    {
        t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + 31]*t_A[t_y*(TILE_SIZE+1) + 31];
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
        
    if(t_x==4)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 4)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  4]*row_data[global_y*(TILE_SIZE+1) + 4];
    }
    __syncthreads();
        
    if(t_x==5)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 5)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  5]*row_data[global_y*(TILE_SIZE+1) + 5];
    }
    __syncthreads();
        
    if(t_x==6)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 6)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  6]*row_data[global_y*(TILE_SIZE+1) + 6];
    }
    __syncthreads();
        
    if(t_x==7)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 7)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  7]*row_data[global_y*(TILE_SIZE+1) + 7];
    }
    __syncthreads();
        
    if(t_x==8)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 8)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  8]*row_data[global_y*(TILE_SIZE+1) + 8];
    }
    __syncthreads();
        
    if(t_x==9)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 9)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  9]*row_data[global_y*(TILE_SIZE+1) + 9];
    }
    __syncthreads();
        
    if(t_x==10)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 10)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  10]*row_data[global_y*(TILE_SIZE+1) + 10];
    }
    __syncthreads();
        
    if(t_x==11)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 11)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  11]*row_data[global_y*(TILE_SIZE+1) + 11];
    }
    __syncthreads();
        
    if(t_x==12)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 12)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  12]*row_data[global_y*(TILE_SIZE+1) + 12];
    }
    __syncthreads();
        
    if(t_x==13)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 13)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  13]*row_data[global_y*(TILE_SIZE+1) + 13];
    }
    __syncthreads();
        
    if(t_x==14)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 14)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  14]*row_data[global_y*(TILE_SIZE+1) + 14];
    }
    __syncthreads();
        
    if(t_x==15)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 15)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  15]*row_data[global_y*(TILE_SIZE+1) + 15];
    }
    __syncthreads();
        
    if(t_x==16)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 16)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  16]*row_data[global_y*(TILE_SIZE+1) + 16];
    }
    __syncthreads();
        
    if(t_x==17)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 17)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  17]*row_data[global_y*(TILE_SIZE+1) + 17];
    }
    __syncthreads();
        
    if(t_x==18)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 18)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  18]*row_data[global_y*(TILE_SIZE+1) + 18];
    }
    __syncthreads();
        
    if(t_x==19)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 19)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  19]*row_data[global_y*(TILE_SIZE+1) + 19];
    }
    __syncthreads();
        
    if(t_x==20)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 20)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  20]*row_data[global_y*(TILE_SIZE+1) + 20];
    }
    __syncthreads();
        
    if(t_x==21)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 21)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  21]*row_data[global_y*(TILE_SIZE+1) + 21];
    }
    __syncthreads();
        
    if(t_x==22)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 22)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  22]*row_data[global_y*(TILE_SIZE+1) + 22];
    }
    __syncthreads();
        
    if(t_x==23)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 23)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  23]*row_data[global_y*(TILE_SIZE+1) + 23];
    }
    __syncthreads();
        
    if(t_x==24)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 24)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  24]*row_data[global_y*(TILE_SIZE+1) + 24];
    }
    __syncthreads();
        
    if(t_x==25)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 25)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  25]*row_data[global_y*(TILE_SIZE+1) + 25];
    }
    __syncthreads();
        
    if(t_x==26)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 26)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  26]*row_data[global_y*(TILE_SIZE+1) + 26];
    }
    __syncthreads();
        
    if(t_x==27)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 27)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  27]*row_data[global_y*(TILE_SIZE+1) + 27];
    }
    __syncthreads();
        
    if(t_x==28)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 28)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  28]*row_data[global_y*(TILE_SIZE+1) + 28];
    }
    __syncthreads();
        
    if(t_x==29)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 29)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  29]*row_data[global_y*(TILE_SIZE+1) + 29];
    }
    __syncthreads();
        
    if(t_x==30)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 30)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  30]*row_data[global_y*(TILE_SIZE+1) + 30];
    }
    __syncthreads();
        
    if(t_x==31)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
    }
    __syncthreads();
    if(t_x > 31)
    {
        row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  31]*row_data[global_y*(TILE_SIZE+1) + 31];
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
        
    valueToSubtract+= row_data[4 + global_y*(TILE_SIZE+1)]*row_data[4 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[5 + global_y*(TILE_SIZE+1)]*row_data[5 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[6 + global_y*(TILE_SIZE+1)]*row_data[6 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[7 + global_y*(TILE_SIZE+1)]*row_data[7 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[8 + global_y*(TILE_SIZE+1)]*row_data[8 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[9 + global_y*(TILE_SIZE+1)]*row_data[9 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[10 + global_y*(TILE_SIZE+1)]*row_data[10 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[11 + global_y*(TILE_SIZE+1)]*row_data[11 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[12 + global_y*(TILE_SIZE+1)]*row_data[12 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[13 + global_y*(TILE_SIZE+1)]*row_data[13 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[14 + global_y*(TILE_SIZE+1)]*row_data[14 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[15 + global_y*(TILE_SIZE+1)]*row_data[15 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[16 + global_y*(TILE_SIZE+1)]*row_data[16 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[17 + global_y*(TILE_SIZE+1)]*row_data[17 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[18 + global_y*(TILE_SIZE+1)]*row_data[18 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[19 + global_y*(TILE_SIZE+1)]*row_data[19 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[20 + global_y*(TILE_SIZE+1)]*row_data[20 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[21 + global_y*(TILE_SIZE+1)]*row_data[21 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[22 + global_y*(TILE_SIZE+1)]*row_data[22 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[23 + global_y*(TILE_SIZE+1)]*row_data[23 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[24 + global_y*(TILE_SIZE+1)]*row_data[24 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[25 + global_y*(TILE_SIZE+1)]*row_data[25 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[26 + global_y*(TILE_SIZE+1)]*row_data[26 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[27 + global_y*(TILE_SIZE+1)]*row_data[27 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[28 + global_y*(TILE_SIZE+1)]*row_data[28 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[29 + global_y*(TILE_SIZE+1)]*row_data[29 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[30 + global_y*(TILE_SIZE+1)]*row_data[30 + global_x*(TILE_SIZE+1)];
        
    valueToSubtract+= row_data[31 + global_y*(TILE_SIZE+1)]*row_data[31 + global_x*(TILE_SIZE+1)];
        
    edit_data[t_y*(TILE_SIZE+1) + t_x]-= valueToSubtract;
    __syncthreads();
}

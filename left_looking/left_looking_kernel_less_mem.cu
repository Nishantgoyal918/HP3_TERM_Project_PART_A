#include "./headers.h"

/*
 * Left looking kernel code (without loop unrolling)
 * INPUT: g_in: pointer of matrix present in Global memory of GPU
 *        N: Matrix size
 * 
 * Stores the lower triangular matrix back in g_in
 */
 __global__ void left_looking_kernel_less_mem(float *g_in, int N)
 {
 
     // extern __shared__ float s_current_panel[4 * TILE_SIZE * TILE_SIZE];
     extern __shared__ float s_current_panel[];
 
 
     // Pointers for accessing shared memory locations
     float *rA1 = NULL;
     float *rA2 = NULL;
     float *rA3 = NULL;
 
     // no of tiles in a column
     int no_of_tiles = (N / TILE_SIZE) + (N % TILE_SIZE != 0); // ceil(N / TILE_SIZE)
 
 
     // i: current panel
     for(int i=0; i<no_of_tiles; i++)
     {
 
         // loading tile(i, i)
         rA1 = &s_current_panel[0];
         load_full(g_in, rA1, i, i, N);
 
 
         for(int j=0; j<no_of_tiles; j++)
         {
 
             if(j >= i)
             {
                 if(j == i)
                 {
                     for(int k=0; k<i; k++)
                     {
 
                         rA2 = &s_current_panel[2 * TILE_SIZE * TILE_SIZE];
                         load_full(g_in, rA2, j, k, N);
 
                         rA1 = &s_current_panel[0];
                         ssyrk_tile(rA1, rA2);
                         __syncthreads();
 
                     }
 
 
                     rA1 = &s_current_panel[0];
                     spotrf_tile(rA1);
                     __syncthreads();
 
                     store_lower(g_in, rA1, i, i, N);
                 }
                 else
                 {
 
                     rA3 = &s_current_panel[1 * TILE_SIZE * TILE_SIZE];
                     load_full(g_in, rA3, j, i, N);
 
                     for(int k=0; k<i; k++)
                     {
 
                         rA1 = &s_current_panel[2 * TILE_SIZE * TILE_SIZE];
                         load_full(g_in, rA1, i, k, N);
 
                         rA2 = &s_current_panel[3 * TILE_SIZE * TILE_SIZE];
                         load_full(g_in, rA1, j, k, N);
 
 
                         sgemm_tile(rA1, rA2, rA3);
                         __syncthreads();
 
                     }
 
 
                     rA1 = &s_current_panel[0];
                     rA2 = &s_current_panel[1 * TILE_SIZE * TILE_SIZE];
 
                     strsm_tile(rA1, rA2);
                     __syncthreads();
 
                     store_full(g_in, rA2, j, i, N);
                 }
 
             }
             else
             {
                 store_zeros(g_in, j, i, N);
             }
             
         }
         
         __syncthreads();
     }
 }
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>       // needed for the function sqrtf()

#define TILE_SIZE 10 // NB // Block SIZE
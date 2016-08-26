 /* Device code. */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "gauss_eliminate.h"

__global__ void gauss_eliminate_kernel(float* U, int *k)
{
	 int tx = threadIdx.x;
	 int ty = threadIdx.y;
	 int col = blockDim.x * blockIdx.x + tx;
	 int row = blockDim.y * blockIdx.y + ty;

	 //int k = k[0];
	 //int i;
	 if(col>k[0] && row ==0)
	 {
		//printf("col: %d,\t", U[MATRIX_SIZE * k[0] + col]);
		//U[MATRIX_SIZE * k[0] + col] = (U[MATRIX_SIZE * k[0] + col] / U[MATRIX_SIZE* k[0] + k[0]]);
		//printf("col: %d\t",U[MATRIX_SIZE * k[0] + col]);
	 }
	// printf("col: %d,\t", U[MATRIX_SIZE *row + col]);
	  __syncthreads();

	  //U[MATRIX_SIZE*k[0] + k[0]] = 1;
	  //if(tx==0)
	  //printf("k: %d, col: %d \n", k[0], col);
	  //__syncthreads();
	 
		/*if((row > k[0]) && (col > k[0]))
		{
			U[MATRIX_SIZE * row + col] = U[MATRIX_SIZE * row + col] - (U[MATRIX_SIZE * row + k[0]] * U[MATRIX_SIZE * k[0] + col]);
			//__syncthreads();
			U[MATRIX_SIZE * row + k[0]] = 0; 
		}*/

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_

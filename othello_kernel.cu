#include <stdio.h>

#define N 10
#define BLOCKSIZE 10

void minmaxCuda(double *max, double *min, double *a, float &time);

__global__ void minmaxKernel(double *max, double *min, double *a) {
	__shared__ double maxtile[BLOCKSIZE];
	__shared__ double mintile[BLOCKSIZE];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	maxtile[tid] = a[i];
	mintile[tid] = a[i];
	__syncthreads();
	
	// strided index and non-divergent branch
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			if (maxtile[tid + s] > maxtile[tid])
				maxtile[tid] = maxtile[tid + s];
			if (mintile[tid + s] < mintile[tid])
				mintile[tid] = mintile[tid + s];
		}
		__syncthreads();
	}
	
	if (tid == 0) {
		max[blockIdx.x] = maxtile[0];
		min[blockIdx.x] = mintile[0];
	}
}

__global__ void finalminmaxKernel(double *max, double *min) {
	__shared__ double maxtile[BLOCKSIZE];
	__shared__ double mintile[BLOCKSIZE];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	maxtile[tid] = max[i];
	mintile[tid] = min[i];
	__syncthreads();
	
	// strided index and non-divergent branch
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			if (maxtile[tid + s] > maxtile[tid])
				maxtile[tid] = maxtile[tid + s];
			if (mintile[tid + s] < mintile[tid])
				mintile[tid] = mintile[tid + s];
		}
		__syncthreads();
	}
	
	if (tid == 0) {
		max[blockIdx.x] = maxtile[0];
		min[blockIdx.x] = mintile[0];
	}
}

void minmaxCuda(double *max, double *min, double *a, float &time)
{

    double *dev_a = 0;
    double *dev_max = 0;
	double *dev_min = 0;
	float milliseconds = 0;

	dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid(N);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaMalloc((void**)&dev_max, N * sizeof(double));


	cudaMalloc((void**)&dev_min, N * sizeof(double));
   
    cudaMalloc((void**)&dev_a, N * N * sizeof(double));

    cudaMemcpy(dev_a, a, N * N * sizeof(double), cudaMemcpyHostToDevice);


	cudaEventRecord(start);
    minmaxKernel<<<dimGrid, dimBlock>>>(dev_max, dev_min, dev_a);
	finalminmaxKernel<<<1, dimBlock>>>(dev_max, dev_min);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaDeviceSynchronize();

    cudaMemcpy(max, dev_max, N * sizeof(double), cudaMemcpyDeviceToHost);


	cudaMemcpy(min, dev_min, N * sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventElapsedTime(&milliseconds, start, stop);
	time = milliseconds;

    cudaFree(dev_max);
	cudaFree(dev_min);
    cudaFree(dev_a);
}


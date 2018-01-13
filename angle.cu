/***************
CS-426-Project-4
ERKAN ÖNAL
21302017
CUDA programming
***************/


#define NOMINMAX
#define PI 3.14159265
#include "cuda_runtime.h"
#include <iostream>
#include <string>
#include <sstream>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <cuda.h>
#include <time.h>

using namespace std;

float getArraySize(char *filename);
void readFile(char *filename, int arrSize, float **vector1, float **vector2);
float *arrayGenerator(float N);
float findAngle(float N, float *vector1, float *vector2);

__global__ void compute(float N, float *d_vector1, float *d_vector2, float *d_vector3)
{
	extern __shared__ float sharedData[];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float tmp = 0;

	while (tid < N) {
		tmp = tmp + d_vector1[tid] * d_vector2[tid];
		tid = tid + blockDim.x * gridDim.x;
	}

	//put your tmp to shared
	sharedData[threadIdx.x] = tmp;

	//synchronize threads
	__syncthreads();

	//reduction code
	int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i)
			sharedData[threadIdx.x] = sharedData[threadIdx.x] + sharedData[threadIdx.x + i];
		__syncthreads();
		i = i / 2;
	}

	//accumulate into final result
	if (threadIdx.x == 0) {
		atomicAdd(d_vector3, sharedData[0]);
	}

	//Nominator Calculated, Now, Calculate Denominator
	//*********************Calculate sqrt of first vector first*********************//
	tid = threadIdx.x + blockIdx.x * blockDim.x;
	tmp = 0;

	while (tid < N) {
		tmp = tmp + powf(d_vector1[tid], 2);
		tid = tid + blockDim.x * gridDim.x;
	}

	//put your tmp to shared
	sharedData[threadIdx.x] = tmp;

	//synchronize threads
	__syncthreads();

	//reduction code
	i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i)
			sharedData[threadIdx.x] = sharedData[threadIdx.x] + sharedData[threadIdx.x + i];
		__syncthreads();
		i = i / 2;
	}

	//accumulate into final result
	if (threadIdx.x == 0) {
		atomicAdd(d_vector3 + 1, sharedData[0]);
	}
	//*********************Calculate sqrt of second vector*********************//
	tid = threadIdx.x + blockIdx.x * blockDim.x;
	tmp = 0;

	while (tid < N) {
		tmp = tmp + powf(d_vector2[tid], 2);
		tid = tid + blockDim.x * gridDim.x;
	}

	//put your tmp to shared
	sharedData[threadIdx.x] = tmp;

	//synchronize threads
	__syncthreads();

	//reduction code
	i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i)
			sharedData[threadIdx.x] = sharedData[threadIdx.x] + sharedData[threadIdx.x + i];
		__syncthreads();
		i = i / 2;
	}

	//accumulate into final result
	if (threadIdx.x == 0) {
		atomicAdd(d_vector3 + 2, sharedData[0]);
	}
}


int main(int argc, char **argv)
{
	if (argc == 3) {
		float CPU_result, GPU_result;

		//To measure time for CPU
		clock_t start, end;
		float time_for_arr_gen, time_for_cpu_func, time_for_host_to_device, time_for_device_to_host, time_for_kernel_exe;

		//To measure time for GPU
		cudaEvent_t start_gpu, stop_gpu;
		cudaEventCreate(&start_gpu);
		cudaEventCreate(&stop_gpu);

		printf("Cuda Works\n");
		//float N = 435090;
		//float N = 20000000;
		float N = atoi(argv[1]);
		//float blocksize = 32;
		float blocksize = atoi(argv[2]);
		float blocksWillBeCreated = (N / blocksize) + 1;
		//define the input/ouput vectors of the host and kernel 
		float *vector1, *vector2, *d_vector1, *d_vector2;
		float *output, *d_output;

		//initialize defined vectors and output
		start = clock();
		vector1 = arrayGenerator(N);
		vector2 = arrayGenerator(N);
		output = (float*)malloc(3 * sizeof(float));
		output[0] = 0; output[1] = 0; output[2] = 0;

		//allocate for device members
		cudaMalloc(&d_vector1, N * sizeof(float));
		cudaMalloc(&d_vector2, N * sizeof(float));
		cudaMalloc(&d_output, 3 * sizeof(float));
		end = clock();
		time_for_arr_gen = ((double)(end - start)) / CLOCKS_PER_SEC;

		//host to device transfer
		start = clock();
		cudaMemcpy(d_vector1, vector1, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vector2, vector2, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_output, output, 3 * sizeof(float), cudaMemcpyHostToDevice);
		end = clock();
		time_for_host_to_device = ((double)(end - start)) / CLOCKS_PER_SEC;

		//run host function and measure its elapsed time
		start = clock();
		CPU_result = findAngle(N, vector1, vector2);
		end = clock();
		time_for_cpu_func = ((double)(end - start)) / CLOCKS_PER_SEC;

		//run kernel function and measure its elapsed time
		start = clock();
		compute << <(int)blocksWillBeCreated, (int)blocksize, (blocksize * sizeof(float)) >> > (N, d_vector1, d_vector2, d_output);

		cudaThreadSynchronize();
		end = clock();
		time_for_kernel_exe = ((double)(end - start)) / CLOCKS_PER_SEC;

		//device to host transfer
		start = clock();
		cudaMemcpy(output, d_output, 3 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();
		end = clock();
		time_for_device_to_host = ((double)(end - start)) / CLOCKS_PER_SEC;

		output[1] = sqrt(output[1]);
		output[2] = sqrt(output[2]);
		float nominator = output[0];
		//printf("Device nominator is: %f\n\n", nominator);
		float denominator = output[1] * output[2];
		GPU_result = nominator / denominator;
		float value = 180.0 / PI;
		GPU_result = atan(GPU_result) * value;

		//float to int
		int NInt = (int)(N);
		int blocksizeInt = (int)(blocksize);
		int blocksWillBeCreatedInt = (int)(blocksWillBeCreated);
		//
		printf("Info\n");
		printf("__________________\n");
		printf("Number of Elements: %d\n", NInt);
		printf("Number of threads per block: %d\n", blocksizeInt);
		printf("Number of blocks will be created: %d\n", blocksWillBeCreatedInt);

		printf("Time\n");
		printf("__________________\n");
		printf("Time for the array generation: %f ms\n", time_for_arr_gen);
		printf("Time for the CPU function: %f ms\n", time_for_cpu_func);
		printf("Time for the Host to Device transfer: %f ms\n", time_for_host_to_device / 1000);
		printf("Time for the kernel execution: %f ms\n", time_for_kernel_exe / 1000);
		printf("Time for the Device to Host transfer: %f\ ms \n", time_for_device_to_host / 1000);
		printf("Total execution time for GPU: %f ms\n", (time_for_host_to_device + time_for_kernel_exe) / 1000);

		printf("Results\n");
		printf("__________________\n");
		printf("CPU result: %.3f\n", CPU_result);
		printf("GPU result: %.3f\n", GPU_result);
		//

		cudaFree(d_vector1);
		cudaFree(d_vector2);
		free(vector1);
		free(vector2);
	}
	else if (argc == 4) {
		//results
		float CPU_result, GPU_result;

		//To measure time for CPU
		clock_t start, end;
		float time_for_arr_gen, time_for_cpu_func, time_for_host_to_device, time_for_device_to_host, time_for_kernel_exe;

		//To measure time for GPU
		cudaEvent_t start_gpu, stop_gpu;
		cudaEventCreate(&start_gpu);
		cudaEventCreate(&stop_gpu);

		printf("Cuda Works\n");

		//read filename
		//char *filename = "data.txt";
		char *filename = argv[3];
		float numOfArraySize = 0;
		numOfArraySize = getArraySize(filename);
		float N = numOfArraySize;

		//float blocksize = 512;
		float blocksize = atoi(argv[2]);
		float blocksWillBeCreated = (N / blocksize) + 1;

		//define the input/ouput vectors of the host and kernel 
		float *vector1, *vector2, *d_vector1, *d_vector2;
		float *output, *d_output;

		//initialize defined vectors and output
		start = clock();
		vector1 = (float*)malloc(N * sizeof(float));
		vector2 = (float*)malloc(N * sizeof(float));
		output = (float*)malloc(3 * sizeof(float));
		readFile(filename, numOfArraySize, &vector1, &vector2);
		output[0] = 0; output[1] = 0; output[2] = 0;

		//allocate for device members
		cudaMalloc(&d_vector1, N * sizeof(float));
		cudaMalloc(&d_vector2, N * sizeof(float));
		cudaMalloc(&d_output, 3 * sizeof(float));
		end = clock();
		time_for_arr_gen = ((double)(end - start)) / CLOCKS_PER_SEC;

		//host to device transfer
		start = clock();
		cudaMemcpy(d_vector1, vector1, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vector2, vector2, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_output, output, 3 * sizeof(float), cudaMemcpyHostToDevice);
		end = clock();
		time_for_host_to_device = ((double)(end - start)) / CLOCKS_PER_SEC;

		//run host function and measure its elapsed time
		start = clock();
		CPU_result = findAngle(N, vector1, vector2);
		end = clock();
		time_for_cpu_func = ((double)(end - start)) / CLOCKS_PER_SEC;

		//run kernel function and measure its elapsed time
		start = clock();
		compute << <(int)((N / blocksize) + 1), (int)blocksize, (blocksize * sizeof(float)) >> > (N, d_vector1, d_vector2, d_output);

		cudaThreadSynchronize();
		end = clock();
		time_for_kernel_exe = ((double)(end - start)) / CLOCKS_PER_SEC;

		//device to host transfer
		start = clock();
		cudaMemcpy(output, d_output, 3 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();
		end = clock();
		time_for_device_to_host = ((double)(end - start)) / CLOCKS_PER_SEC;

		output[1] = sqrt(output[1]);
		output[2] = sqrt(output[2]);
		float nominator = output[0];
		float denominator = output[1] * output[2];
		GPU_result = nominator / denominator;
		float value = 180.0 / PI;
		GPU_result = atan(GPU_result) * value;

		//float to int
		int NInt = (int)(N);
		int blocksizeInt = (int)(blocksize);
		int blocksWillBeCreatedInt = (int)(blocksWillBeCreated);
		//
		printf("Info\n");
		printf("__________________\n");
		printf("Number of Elements: %d\n", NInt);
		printf("Number of threads per block: %d\n", blocksizeInt);
		printf("Number of blocks will be created: %d\n", blocksWillBeCreatedInt);

		printf("Time\n");
		printf("__________________\n");
		printf("Time for the array generation: %f ms\n", time_for_arr_gen);
		printf("Time for the CPU function: %f ms\n", time_for_cpu_func);
		printf("Time for the Host to Device transfer: %f ms\n", time_for_host_to_device / 1000);
		printf("Time for the kernel execution: %f ms\n", time_for_kernel_exe / 1000);
		printf("Time for the Device to Host transfer: %f\ ms \n", time_for_device_to_host / 1000);
		printf("Total execution time for GPU: %f ms\n", (time_for_host_to_device + time_for_kernel_exe) / 1000);

		printf("Results\n");
		printf("__________________\n");
		printf("CPU result: %.3f\n", CPU_result);
		printf("GPU result: %.3f\n", GPU_result);
		//

		cudaFree(d_vector1);
		cudaFree(d_vector2);
		free(vector1);
		free(vector2);
	}
	else {
		printf("Invalid number of arguements");
	}

	return 0;
}

float *arrayGenerator(float N) {
	if (N < 0)
		return NULL;
	float *vector = (float*)malloc(N * sizeof(float));
	for (int i = 0; i < N; i++) {
		vector[i] = rand() % 20 - 20;;
	}
	return vector;
}

float getArraySize(char *filename) {
	float numOfArraySize = 0;
	FILE* file = fopen(filename, "r");
	fscanf(file, "%f", &numOfArraySize);
	fclose(file);

	return numOfArraySize;
}

void readFile(char *filename, int arrSize, float **vector1, float **vector2) {
	int a = 0;
	FILE* file = fopen(filename, "r");
	fscanf(file, "%d", &a);

	int x = 0;
	int i = 0;
	int j = 0;
	while (!feof(file)) {
		fscanf(file, "%d", &x);
		if (i < arrSize) {
			(*vector1)[i] = x;
		}
		if (i >= arrSize && i < 2 * arrSize) {
			(*vector2)[j] = x;
			j++;
		}
		i++;
	}
	fclose(file);
}
float findAngle(float N, float *vector1, float *vector2) {
	float nominator = 0;
	float length1 = 0;
	float length2 = 0;
	float denominator = 0;
	float result = 0;
	float value = 180.0 / PI;

	for (int i = 0; i < N; i++) {
		//printf("vector1[i]: %d and vector2[i]: %d\n", vector1[i], vector2[i]);
		nominator = nominator + vector1[i] * vector2[i];
	}
	//printf("Host nominator: %f\n", nominator);
	for (int i = 0; i < N; i++) {
		length1 = length1 + pow(vector1[i], 2);
		length2 = length2 + pow(vector2[i], 2);
	}
	length1 = sqrt(length1);
	length2 = sqrt(length2);

	//printf("serial result length1: %f\n", length1);
	//printf("serial result length2: %f\n", length2);

	denominator = length1 * length2;
	//printf("Denominator: %f\n", denominator);

	result = nominator / denominator;
	result = atan(result) * value;

	return result;
}
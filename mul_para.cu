#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define N (1024)

__global__ void inc(int *s, int *d, int len)
{
	int i,j;
	int part;		// 各スレッドが担当するデータの個数
	int idx_start, idx_end;	// 各スレッドの担当範囲
	part = len / (gridDim.x * blockDim.x);	// blockDim:1ブロック中のスレッド数
	idx_start = part * blockDim.x * blockIdx.x + threadIdx.x;	// threadIdx：自分のスレッド番号
	idx_end   = idx_start + part;
	for (j=0; j<1000000; j++) {
	for (i = idx_start; i < idx_end; i++)
		s[i] *= d[i];
		s[i] /= 3;
	}
	return;
}

int dat0[N], dat1[N];
int main(int argc, char *argv[])
{
	bool dout;
	int i;
	int c_time;
	int *s, *d;
	size_t array_size;

	if (argc > 1)
		dout = false;
	else	dout = true;
	for (i=0; i<N;i++) {
		dat0[i] = i+100;
		dat1[i] = N-i;
	}
	if (dout) {
		printf("input:");
		for (i=0; i<N;i++)
			printf("[%d*%d]", dat0[i], dat1[i]);
		printf("\n");
	}
	array_size = sizeof(int) * N;
	cudaMalloc((void **)&s, array_size);
	cudaMalloc((void **)&d, array_size);

	cudaMemcpy(s, dat0, array_size,
			cudaMemcpyHostToDevice);
	cudaMemcpy(d, dat1, array_size,
			cudaMemcpyHostToDevice);
	c_time = (int)clock();
	inc<<<32, 32>>>(s, d, N);
	cudaDeviceSynchronize();
	c_time = (int)clock() - c_time;
	cudaMemcpy(dat0, s, array_size,
		cudaMemcpyDeviceToHost);
	cudaMemcpy(dat1, d, array_size,
		cudaMemcpyDeviceToHost);
	if (dout) {
		printf("output:");
		for (i=0; i<N; i++)
			printf("%d ", dat0[i]);
		printf("\n");
	}
	printf("Time:- %d\n",c_time);
	return 0;
}

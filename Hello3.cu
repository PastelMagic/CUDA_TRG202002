//----------------------------------------
//- Hello CUDA
//- マルチストリームバージョン
//----------------------------------------
//
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define DATLEN 16
#define NSTREAM 4
int Buf[NSTREAM][DATLEN];
int *Data;
int *GPUMem[NSTREAM];

//----------------------------------------
//- GPUで実行される部分
//- 関数名の前に__global__を付ける
//- 引数は呼び出し側で自由に設定できる
//----------------------------------------
//
__global__ void GPUadd(int *src, int len, int dat)
{
	int threads,thnum;
	int start, size;
	start = gridDim.x;
	threads = gridDim.x * blockDim.x;	// 総スレッド数
	size = len / threads;			// １スレッドの担当データ数
	thnum = blockDim.x * blockIdx.x + threadIdx.x;	// このスレッドの通し番号
	start = size * thnum;			// このスレッドが担当する先頭位置

	for (int i = 0; i < size; i++)		// 耐えられたデータ長分だけ
		src[start++] += dat;		// 配列にdat値を加算
	return;
}

//----------------------------------------
//- 配列データ表示
//----------------------------------------
//
void DispData(const char *s, int *dat)
{
	printf("%s\n",s);			// データのキャプション
	for (int j=0; j<NSTREAM; j++) {
		for (int i=0; i<DATLEN; i++)		// 配列データサイズ分全部表示
			printf("%02d ",*dat++);		// 表示して
		printf("\n");				// 最後に改行しておく
	}
	printf("\n");				// 最後に改行しておく
}
//----------------------------------------
//- メイン
//----------------------------------------
//
int main(int argc, char *argv[])
{
	int i;
	size_t DataBytes;

	cudaStream_t Stream[NSTREAM];

	printf("[%s]\n",argv[0]);
	printf("Welcom to CUDA!\n");			// ようこそ！

	for (int i=0; i<NSTREAM; i++)
		cudaStreamCreate(&Stream[i]);
	
	DataBytes = sizeof(int) * DATLEN;		// 配列データの総バイト数を計算して
	Data = Buf[0];

#ifndef SINGLE
	cudaMallocHost(&Data, DataBytes*NSTREAM);	// GPUとの共有メモリ領域から転送サイズ分を確保
#else
	Data =(int*) malloc(DataBytes*NSTREAM);
#endif

	for (int j=0; j<NSTREAM; j++) {
		for (i=0; i<DATLEN;i++)				// 初期値は10からのインクリメントデータにした
			*(Data+j*DATLEN+i) = i+10+j;				// （別になんでも良いのだけど）
	}
	DispData("GPU IN :-", Data);			// とりあえず中身を表示しておく


	for (int i=0; i<NSTREAM; i++)
		cudaMalloc((void **)&GPUMem[i], DataBytes);	// GPUとの共有メモリ領域から転送サイズ分を確保

	int c_time = clock();
	for (int j=0; j<10000; j++) {
		for (int i=0; i<NSTREAM; i++) {
#ifndef SINGLE
			cudaMemcpyAsync(GPUMem[i], Data+i*DATLEN,			// Data[]を共有メモリにコピー
				DataBytes,cudaMemcpyHostToDevice,Stream[i]);
#else
			cudaMemcpy(GPUMem[i], Data+i*DATLEN,			// Data[]を共有メモリにコピー
				DataBytes,cudaMemcpyHostToDevice);
#endif

#ifndef SINGLE
			GPUadd<<<2, 4,0, Stream[i]>>>(GPUMem[i], DATLEN, 3);	// マルチストリームで実行
#else
			GPUadd<<<2, 4,0>>>(GPUMem[i], DATLEN, 3);	// シングルストリームで実行
#endif
	
#ifndef SINGLE
			cudaMemcpyAsync(Data+i*DATLEN, GPUMem[i], DataBytes,		// Stream[i]の担当分のデータをコピー
				cudaMemcpyDeviceToHost, Stream[i]);
#else
			cudaMemcpy(Data+i*DATLEN, GPUMem[i], DataBytes,		// 完了してから共有メモリからData[]にコピー
				cudaMemcpyDeviceToHost);
#endif
		}
#ifdef SYNC
		for (int i=0; i<NSTREAM; i++)
			cudaStreamSynchronize(Stream[i]);	// 同期したければ
#endif
	}
	c_time = clock()-c_time;
	printf("Time:-%d\n",c_time);

	DispData("GPU OUT:-", Data);			// 中身を表示

	printf("Congraturations!\n");			// おめでとうございます！

	return 0;
}

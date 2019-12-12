//----------------------------------------
//- Hello CUDA
//- 
//----------------------------------------
//
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define DATLEN 16
int Data[DATLEN];
int *GPUMem;

//----------------------------------------
//- GPUで実行される部分
//- 関数名の前に__global__を付ける
//- 引数は呼び出し側で自由に設定できる
//----------------------------------------
//
__global__ void GPUadd(int *src, int len, int dat)
{
	for (int i = 0; i < len; i++)		// 耐えられたデータ長分だけ
		src[i] += dat;			// 配列にdat値を加算
	return;
}

//----------------------------------------
//- 配列データ表示
//----------------------------------------
//
void DispData(const char *s, int *dat)
{
	printf("%s ",s);			// データのキャプション
	for (int i=0; i<DATLEN; i++)		// 配列データサイズ分全部表示
		printf("%02d ",*dat++);		// 表示して
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
	printf("Welcom to CUDA!\n");			// ようこそ！
	for (i=0; i<DATLEN;i++)				// 初期値は10からのインクリメントデータにした
		Data[i] = i+10;				// （別になんでも良いのだけど）
	DispData("GPU IN :-", Data);			// とりあえず中身を表示しておく

	DataBytes = sizeof(int) * DATLEN;		// 配列データの総バイト数を計算して
	cudaMalloc((void **)&GPUMem, DataBytes);	// GPUとの共有メモリ領域から転送サイズ分を確保
	cudaMemcpy(GPUMem, Data,			// Data[]を共有メモリにコピー
		DataBytes,cudaMemcpyHostToDevice);

	GPUadd<<<1, 1>>>(GPUMem, DATLEN, 3);		// GPUadd()関数をGPUで実行

	cudaMemcpy(Data, GPUMem, DataBytes,		// 完了してから共有メモリからData[]にコピー
		cudaMemcpyDeviceToHost);
	DispData("GPU OUT:-", Data);			// 中身を表示

	printf("Congraturations!\n");			// おめでとうございます！

	return 0;
}

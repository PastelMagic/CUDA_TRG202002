//-------------------------------------------------
//- GPU演算テスト（ユニファイドメモリ版）
//-　　物体間、及び中心位置から距離の自乗に比例する
//-    ような吸引力が働くとして動かしてみた
//- コンパイル時のオプション
//- nvcc -o xxx xxx.cu -lX11 -lm
//- CPUと同じコードをGPUを１スレッドで動かしたい時は
//-   -D USE_GPU_SINGLEを付加
//- CPUで動かしたい時は
//-   -D USE_CPUを付加
//- ex.
//- nvcc -o xxx xxx.cu -D USE_CPU -lX11 -lm
//-------------------------------------------------
//

// #define USE_CPU
// #define USE_GPU_SINGLE

#include <stdio.h>
#include <math.h>
#include <X11/Xlib.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WIN_W     600	// ウインドウの横幅
#define WIN_H     600	// ウインドウの縦幅
#define CENT_X    (WIN_W>>1)	// X方向の中央位置
#define CENT_Y    (WIN_H>>1)	// Y方向の中央位置
#define WIN_X     100	// ウインドウの左上のX座標
#define WIN_Y     100	// ウインドウの左上のY座法
#define BORD_W    2	// ボーダ幅

#define	OBJN	10	// ボールの数（2^N個)
#define	OBJS	(1<<OBJN)	// ボールの数（N個）

unsigned long  bcol;
unsigned long  black,white; /* 黒と白のピクセル値 */


#ifndef USE_CPU
__device__ __managed__ double *posX, *posY;	//pos[2][OBJS];
__device__ __managed__ double *spdX, *spdY;	//spd[2][OBJS];
__device__ __managed__ double wt[OBJS];		//wt[OBJS];
__device__ __managed__ int Nobj;	// ユニファイドメモリ上の変数として定義

#else
double *posX, *posY;	//pos[2][OBJS];
double *spdX, *spdY;	//spd[2][OBJS];
double wt[OBJS];	//wt[OBJS];
int Nobj;

#endif


#ifndef USE_CPU
__global__ void GPUmove()
{
	int Start, End, Size;
	int NThreads;
	int thread;

	NThreads = gridDim.x * blockDim.x;	// 総スレッド数を算出
	if (NThreads >= Nobj)			// データ数よりスレッド数の方が多い
		NThreads = Nobj;		// データ数に合わせる
	Size = Nobj / NThreads;			// 1スレッド当たりの担当数（割り切れる前提）
	thread = blockDim.x * blockIdx.x + threadIdx.x;	// 自分のスレッド番号
	if (thread < NThreads) {		// スレッド数未満なら担当分あり
		Start = thread * Size;
		End = Start + Size;
		for (int i=Start; i<End; i++) {
			double fx,fy;
			// 誤差蓄積などで早期に飛び散っていかないように
			// 中心に寄せる力を加えておく
			fx= -posX[i];
			fx *= fx*fx/512;
			fy= -posY[i];
			fy *= fy*fy/512;
	
			int targ;		//　引き合い力を計算する相手
			for (int j=1; j<Nobj; j++) {		// 自分との関係は計算しなくていい
				targ = (i+j) & (Nobj-1);	// 相手を決めて
				double dfX = posX[targ] - posX[i];	// X方向の距離
				double dfY = posY[targ] - posY[i];	// Y方向の距離
				double dist = sqrt(dfX*dfX + dfY*dfY);	// 直線距離
				fx += dfX*dist/1024;	// 距離の自乗に比例したX方向の力
				fy += dfY*dist/1024;	// 距離の自乗に比例したY方向の力
			}
			spdX[i] += (fx/wt[i]);		// X方向の加速度に応じて速度を加算
			spdY[i] += (fy/wt[i]);		// Y方向の加速度に応じて速度を加算
		}
		for (int i=Start; i<End; i++) {		// すべての速度を算出したので
			posX[i] += spdX[i];		// X方向の速度分だけ座標移動
			posY[i] += spdY[i];		// Y方向の速度分だけ座標移動
		}
	}
}
#endif

#ifdef USE_GPU_SINGLE
__global__ void move()
#else
void move()
#endif
{
	for (int i=0; i<Nobj; i++) {
		double fx,fy;
		// 誤差蓄積などで早期に飛び散っていかないように
		// 中心に寄せる力を加えておく
		fx= -posX[i];
		fx *= fx*fx/512;
		fy= -posY[i];
		fy *= fy*fy/512;

		int targ;		//　引き合い力を計算する相手
		for (int j=1; j<Nobj; j++) {		// 自分との関係は計算しなくていい
			targ = (i+j) & (Nobj-1);	// 相手を決めて
			double dfX = posX[targ] - posX[i];	// X方向の距離
			double dfY = posY[targ] - posY[i];	// Y方向の距離
			double dist = sqrt(dfX*dfX + dfY*dfY);	// 直線距離
			fx += dfX*dist/1024;	// 距離の自乗に比例したX方向の力
			fy += dfY*dist/1024;	// 距離の自乗に比例したY方向の力
		}
		spdX[i] += (fx/wt[i]);		// X方向の加速度に応じて速度を加算
		spdY[i] += (fy/wt[i]);		// Y方向の加速度に応じて速度を加算
	}
	for (int i=0; i<Nobj; i++) {		// すべての速度を算出したので
		posX[i] += spdX[i];		// X方向の速度分だけ座標移動
		posY[i] += spdY[i];		// Y方向の速度分だけ座標移動
	}
}


//-----------------------------------
//-- 移動処理呼び出しディスパッチャ
//-----------------------------------
//
int DataBytes = sizeof(double) * OBJS;		// 配列データの総バイト数を計算
						// 全部Doubleなので共通で使う
void DispatchMove()
{
#ifndef USE_CPU
  #ifndef USE_GPU_SINGLE	// シングルスレッドじゃない（マルチスレッド）
	GPUmove<<<32, 32>>>();
	cudaDeviceSynchronize();	// GPUの演算終了を待つ
  #else				// シングルスレッド
	move<<<1, 1>>>();
//	GPUmove<<<1, 1>>>(posX, posY, spdX, spdY, wt);
	cudaDeviceSynchronize();	// GPUの演算終了を待つ
  #endif
#else				// CPUで頑張る
	move();
#endif
}


//-----------------------------------
//-- 配列等の初期化
//-----------------------------------
//
void init(void)
{
	Nobj = OBJS;
#ifndef USE_CPU
	cudaMallocManaged(&posX, DataBytes);	// ユニファイドメモリ確保
	cudaMallocManaged(&posY, DataBytes);
	cudaMallocManaged(&spdX, DataBytes);
	cudaMallocManaged(&spdY, DataBytes);
#else
	posX = (double *)malloc(DataBytes);	// CPU使用時はmalloc()で済ませる
	posY = (double *)malloc(DataBytes);
	spdX = (double *)malloc(DataBytes);
	spdY = (double *)malloc(DataBytes);
#endif
	for (int i=0; i<OBJS; i++) {
		spdX[i] = 0;		// 速度のX方向初期値はゼロ
		spdY[i] = 0;;		// 速度のY方向初期値はゼロ
		wt[i] = OBJS*1000+(i & ~0x3)*800;	// 4個おきに重さを変化させてバランスさせる
							// 数が増えると掛かる力も増加しがちなので、
							// 重さも増えるようにした
	}
	// オブジェクトの初期位置（結構適当にやってる）
	double p = (double)WIN_W/(double)OBJS;
	double stp = p;
	double offset = WIN_W/64/OBJS;
	for (int i=0; i<OBJS; i+= 4, p+=stp) {
		offset = i*i*WIN_W/OBJS/OBJS/8;
		posX[i] = p;
		posY[i] = offset;

		posX[i+1]= -offset;
		posY[i+1] = p;

		posX[i+2] = -p;
		posY[i+2] = -offset;

		posX[i+3] = offset;
		posY[i+3] = -p;
	}
}


//-----------------------------------
//-- GPU上に確保したメモリ領域解放
//-----------------------------------
void release()
{
#ifndef USE_CPU
	cudaFree(posX);
	cudaFree(posY);
	cudaFree(spdX);
	cudaFree(spdY);
#else
	free(posX);
	free(posY);
	free(spdX);
	free(spdY);
#endif
}

//-----------------------------------
//-- 指定位置に四角を描く
//-----------------------------------
//
static void draw_dot(Display *dpy, Window win, GC gc,
  unsigned int x, unsigned int y, unsigned long color)
{
	XSetForeground( dpy, gc, color );
//	XFillArc( dpy, win, gc,x, y, 5, 5, 0, 360*64);
	XFillRectangle(dpy, win, gc, x-1, y-1, 2, 2);
}

//-----------------------------------
//-- オブジェクトの描画
//-----------------------------------
//
void draw(Display *dpy, Window win, GC gc)
{
	XSetForeground( dpy, gc, black );
	XFillRectangle(dpy, win, gc, 0, 0, WIN_W, WIN_H);
	for (int i=0; i<OBJS; i++) {
		int x = (int)posX[i]+CENT_X;
		int y = (int)posY[i]+CENT_Y;
		draw_dot(dpy, win, gc, x, y, bcol);
	}
}

int main( void )
{
	Display*       dpy;         /* ディスプレイ */
	Window         root;        /* ルートウィンドウ */
	Window         win;         /* 表示するウィンドウ */
	int            screen;      /* スクリーン */
	GC             gc;          /* グラフィックスコンテキスト */
	XEvent         evt;         /* イベント構造体 */
	Colormap       cmap;        /* カラーマップ */
	XColor         color, exact;

	init();

	/* Xサーバと接続する */
	dpy = XOpenDisplay( "" );

	/* ディスプレイ変数の取得 */
	root   = DefaultRootWindow( dpy );
	screen = DefaultScreen( dpy );
	/* XAllocNamedColor() のためにカラーマップを取得 */
	cmap   = DefaultColormap( dpy, screen );

	white  = WhitePixel( dpy, screen );
	black  = BlackPixel( dpy, screen );
	XAllocNamedColor( dpy, cmap, "MistyRose", &color, &exact );
	bcol = color.pixel;

	// ウインドウの作成
	win = XCreateSimpleWindow( dpy, root,
	   WIN_X, WIN_Y, WIN_W, WIN_H, BORD_W, white, black);
	// グラフィックコンテキスト作成
	gc = XCreateGC( dpy, win, 0, NULL );
	// 再描画と、キーボード押下（終了させるのに使っている）イベントを取得
	XSelectInput( dpy, win, KeyPressMask | ExposureMask );
	// ウインドウ表示
	XMapWindow( dpy, win );

	int LoopCount = 0;
	Bool terminate = False;		// プログラム終了フラグ
	while( !terminate) {
		if (XEventsQueued(dpy, QueuedAfterFlush) != 0) {	// イベントが入っている
			XNextEvent( dpy, &evt );	// イベントをとり出す
			switch( evt.type ) {
				case Expose:		// 再描画が必要になった
					if ( evt.xexpose.count == 0 ) {
						draw( dpy, win, gc);
					}
					break;
				case KeyPress:		// キーが押されたら
					XFreeGC( dpy, gc );	// コンテキストを開放して
					XDestroyWindow( dpy, win );	// ウインドウを破棄
					XCloseDisplay( dpy );		// 表示終了
					terminate = True;		// 終了する
					break;
				default:
					break;
			}
		}
		if (!terminate) {
			DispatchMove();		// 移動処理を呼び出す
			draw(dpy, win, gc);	// 描画
//			usleep(10*1000);	// 速すぎる時にちょっと待たせたいならここ
			LoopCount++;
		}
	}
	printf("Loop:-%d\n",LoopCount);
	release();
	return 0;
}

//---------------------------------
//- cc -o gsim gsim.c -lX11 -lm
//---------------------------------
//
#include <stdio.h>
#include <math.h>
#include <X11/Xlib.h>
#include <unistd.h>
#define WIN_W     600  /* ウィンドウの幅   */
#define WIN_H     600  /* ウィンドウの高さ */
#define CENT_X    (WIN_W>>1)
#define CENT_Y    (WIN_H>>1)
#define WIN_X     100  /* ウィンドウ表示位置(X) */
#define WIN_Y     100  /* ウィンドウ表示位置(Y) */
#define BORDER    2    /* ボーダの幅      */

#define	OBJN	10		// ボールの数（2^2個)
#define	OBJS	(1<<OBJN)	// ボールの数（2^2個）

unsigned long  bcol;
unsigned long  CBlack,CWhite; /* 黒と白のピクセル値 */


double pos[OBJS][3];
double spd[OBJS][3];
double wt[OBJS];
int col[OBJS];

void init(void)
{
	for (int i=0; i<OBJS; i++) {
		spd[i][0] = 0;
		spd[i][1] = 0;
		spd[i][2] = 0;
		wt[i] = OBJS*1000+(i & ~0x3)*800;
	}
	double p = (double)WIN_W/(double)OBJS;
	double stp = p;
	double offset = WIN_W/64/OBJS;
	for (int i=0; i<OBJS; i+= 4, p+=stp) {
		offset = i*i*WIN_W/OBJS/OBJS/8;
		pos[i][0] = p;
		pos[i][1] = offset;
		pos[i][2] = 0;

		pos[i+1][0] = -offset;
		pos[i+1][1] = p;
		pos[i+1][2] = 0;

		pos[i+2][0] = -p;
		pos[i+2][1] = -offset;
		pos[i+2][2] = 0;

		pos[i+3][0] = offset;
		pos[i+3][1] = -p;
		pos[i+3][2] = 0;

		offset++;

	}
}

static void draw_dot(Display *dsp, Window win, GC gc,
  unsigned int x, unsigned int y, unsigned long color)
{
	XSetForeground( dsp, gc, color );
//	XFillArc( dsp, win, gc,x, y, 5, 5, 0, 360*64);
	XFillRectangle(dsp, win, gc, x-1, y-1, 2, 2);
}

static void move(Display *dsp, Window win, GC gc)
{
	int x,y,z;
	for (int i=0; i<OBJS; i++) {
		double fx,fy,fz;
		fx= -pos[i][0];
		fx *= fx*fx/512;
		fy= -pos[i][1]; 
		fy *= fy*fy/512;
		fz= -pos[i][2]; 
		fz *= fz*fz/512;
		int targ;
		for (int j=1; j<OBJS; j++) {
			targ = (i+j) & (OBJS-1);
			double dist = 0;
			double df[3];
			for (int k=0; k<3; k++) {
				double d = df[k] = pos[targ][k]-pos[i][k];
				dist += d*d;
			}
			dist = sqrt(dist);
			fx += df[0]*dist/1024;
			fy += df[1]*dist/1024;
			fz += df[2]*dist/1024;
		}
		spd[i][0] += (fx/wt[i]);
		spd[i][1] += (fy/wt[i]);
		spd[i][2] += (fz/wt[i]);
	}
	XSetForeground( dsp, gc, CBlack );
	XFillRectangle(dsp, win, gc, 0, 0, WIN_W, WIN_H);
	for (int i=0; i<OBJS; i++) {
		pos[i][0] += spd[i][0];
		pos[i][1] += spd[i][1];
		pos[i][2] += spd[i][2];
		x = (int)pos[i][0]+CENT_X;
		y = (int)pos[i][1]+CENT_Y;
		z = (int)pos[i][2];
		draw_dot(dsp, win, gc, x, y, bcol);

	}

}

int cx, cy, dx, dy, px, py;

int main( void )
{
	Display*       dsp;         /* ディスプレイ */
	Window         root;        /* ルートウィンドウ */
	Window         win;         /* 表示するウィンドウ */
	int            screen;      /* スクリーン */
	GC             gc;          /* グラフィックスコンテキスト */
	XEvent         evt;         /* イベント構造体 */
	Colormap       cmap;        /* カラーマップ */
	XColor         color, exact;
//	unsigned long  green, red;  /* 緑と赤のピクセル値 */
	cx = 10;
	cy = 10;
	px = cx;
	py = cy;
	dx = 1;
	dy = 1;
	/* Xサーバと接続する */
	dsp = XOpenDisplay( "" );

	/* ディスプレイ変数の取得 */
	root   = DefaultRootWindow( dsp );
	screen = DefaultScreen( dsp );
	/* XAllocNamedColor() のためにカラーマップを取得 */
	cmap   = DefaultColormap( dsp, screen );
	/* 白、黒のピクセル値を取得 */
	CWhite  = WhitePixel( dsp, screen );
	CBlack  = BlackPixel( dsp, screen );
	/* 緑と赤のピクセル値を取得 */
//	XAllocNamedColor( dsp, cmap, "green", &color, &exact );
//	green = color.pixel;
//	XAllocNamedColor( dsp, cmap, "red", &color, &exact );
//	red = color.pixel;
//	XAllocNamedColor( dsp, cmap, "blue", &color, &exact );
//	blue = color.pixel;
	XAllocNamedColor( dsp, cmap, "MistyRose", &color, &exact );
	bcol = color.pixel;

	// ウインドウ作成
	win = XCreateSimpleWindow( dsp, root,
	   WIN_X, WIN_Y, WIN_W, WIN_H, BORDER, CWhite, CBlack);
	// グラフィックコンテキスト作成
	gc = XCreateGC( dsp, win, 0, NULL );
	// Xサーバーからキー入力と再描画イベントを取得
	XSelectInput( dsp, win, KeyPressMask | ExposureMask );
	// ウインドウをマッピング（この時点で描画される）
	XMapWindow( dsp, win );

	// オブジェクトの初期化
	init();

	// 以下無限ループ
	while(1) {
		if (XEventsQueued(dsp, QueuedAfterFlush) != 0) {
			XNextEvent( dsp, &evt );
			switch( evt.type ) {
				case Expose:
					if ( evt.xexpose.count == 0 ) {
						move( dsp, win, gc);
					}
					break;
				case KeyPress:
					/* リソースの解放 */
					XFreeGC( dsp, gc );
					XDestroyWindow( dsp, win );
					XCloseDisplay( dsp );
					return 0;
			}
		}
		move(dsp, win, gc);
//		usleep(10*1000);
	}
}


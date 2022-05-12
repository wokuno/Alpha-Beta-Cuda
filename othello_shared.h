#ifndef _OTHELLO_SHARED_H_
#define _OTHELLO_SHARED_H_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_CONSTANT __constant__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_CONSTANT
#endif 

#include <stdio.h>

#define BOARDSIZE 100

 
#define EMPTY 0   
#define BLACK 1
#define WHITE 2
#define OUTER 3

#define WIN 99999999
#define LOSS -99999999

CUDA_CONSTANT const int ALLDIRECTIONS[8]={-11, -10, -9, -1, 1, 9, 10, 11};

CUDA_CALLABLE_MEMBER char nameof (int piece);
CUDA_CALLABLE_MEMBER int opponent (int player);
CUDA_CALLABLE_MEMBER int * copyboard (int * board);
CUDA_CALLABLE_MEMBER int * initialboard (void);
CUDA_CALLABLE_MEMBER int count (int player, int * board);
CUDA_CALLABLE_MEMBER void printboard (int * board);
CUDA_CALLABLE_MEMBER int validp (int move);
CUDA_CALLABLE_MEMBER int findbracketingpiece(int square, int player, int * board, int dir);
CUDA_CALLABLE_MEMBER int wouldflip (int move, int player, int * board, int dir);
CUDA_CALLABLE_MEMBER int legalp (int move, int player, int * board);
CUDA_CALLABLE_MEMBER void makeflips (int move, int player, int * board, int dir);
CUDA_CALLABLE_MEMBER void makemove (int move, int player, int * board);
CUDA_CALLABLE_MEMBER int anylegalmove (int player, int * board);
CUDA_CALLABLE_MEMBER int nexttoplay (int * board, int previousplayer, int printflag);
CUDA_CALLABLE_MEMBER int cudanexttoplay (int * board, int previousplayer, int printflag);

CUDA_CALLABLE_MEMBER int * legalmoves (int player, int * board);

CUDA_CALLABLE_MEMBER int diffeval (int player, int * board);

typedef int (* fpc) (int, int *);

#endif
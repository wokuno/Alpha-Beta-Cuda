#ifndef _ALPHA_BETA_H_
#define _ALPHA_BETA_H_

typedef struct nud {
    int * board;
    int move;
    int player;
    int alpha;
    int beta;
    struct nud *parent;
} Node;

#endif
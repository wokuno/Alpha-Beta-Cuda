#ifndef _NODE_H_
#define _NODE_H_

typedef struct {
    int * board;
    int move;
    int player;
    int maximizer;
    int alpha;
} Node;

#endif
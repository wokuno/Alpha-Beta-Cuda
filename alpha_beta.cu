#include "alpha_beta.h"

__device__ void cudaSearch(Node *node, int player, int maximizer, int ply) {
    if (ply == 0){
        node->alpha = diffeval(maximizer, node->board);
        node->beta = diffeval(maximizer, node->board);
        return;
    }

    int * moves = legalmoves(node->player, node->board);
    if(moves[0] == 0) return;

    for (int i = 1; i < moves[0]; i++) {
        int * newboard = copyboard(node->board);
        int move = moves[i];
        makemove(moves[1], opponent(node->player), newboard);
        int ntm = nexttoplay(newboard, opponent(node->player), 0);
        if (ntm == 0){
            node->alpha = diffeval(node->player, node->board);
            node->beta = diffeval(node->player, node->board);
            return;
        }
        ntm = cudanexttoplay(newboard, opponent(node->player), 0);
    
        // makemove(move, node->player, newboard);
    
        Node *newNode = node;
        newNode->move = move;
        newNode->player = ntm;
        newNode->alpha = node->alpha;
        newNode->beta = node->beta;
        newNode->board = newboard;
        newNode->parent = node;

        // search child
        cudaSearch(newNode, ntm, maximizer, ply - 1);

        if (player == ntm) {
            node->beta = min(node->beta, newNode->alpha);
        } 
        if (opponent(player) == ntm){
            node->alpha = max(node->alpha, newNode->beta);
        }

        if (node->alpha >= node->beta) {
            return;
        }
        free(newNode);
    }
}

__global__
void cudaTreeKernel(int * moves, int * board, int * values, int player, int maximizer,
    int alpha, int beta, int ply) {
    // only one thread does high-level tasks
    if (threadIdx.x == 0) {
        // make one new node per block
        if(moves[0] == 0) return;
        int move = moves[blockIdx.x];
        int * newboard = copyboard(board);
        makemove(move, player, newboard);
        int ntm = cudanexttoplay(newboard, player, 0);
        
        Node *newNode = new Node;
        newNode->move = move;
        newNode->player = ntm;
        newNode->alpha = alpha;
        newNode->beta = beta;
        newNode->board = newboard;

        cudaSearch(newNode, player, maximizer, ply);

        // update the values we care about - if the parent node is a maximizing node, 
        // it cares about the child alpha values
        if (player == maximizer) {
            values[blockIdx.x] = newNode->beta;
        } 
        if (opponent(player) == maximizer){
            values[blockIdx.x] = newNode->alpha;
        }
        free(newNode);
    }
}

void cudaMinMaxKernel(int * moves, int * board, int *values, int player, int maximizer, int alpha, int beta, int numMoves, int ply) {

    cudaTreeKernel<<<numMoves, 32>>>(moves, board, values, player, maximizer, alpha, beta, ply);
}

int search(Node *node, int maximizer, int ply) {
    // Do not search any deeper
    if (ply == 0){
        node->alpha = diffeval(maximizer, node->board);
        node->beta = diffeval(maximizer, node->board);
        return NULL;
    }

    // make copy of board and find moves
    int * newboard = copyboard(node->board);
    int * moves = legalmoves(node->player, node->board);
    makemove(moves[1], opponent(node->player), newboard);
    int ntm = cudanexttoplay(newboard, node->player, 0);
    
    Node *newNode = node;
    newNode->move = moves[1];
    newNode->player = ntm;
    newNode->alpha = node->alpha;
    newNode->beta = node->beta;
    newNode->board = newboard;
    newNode->parent = node;

    int best = search(newNode,maximizer, ply - 1);

    int *values;

    values = (int *)calloc(moves[0], sizeof(int));

    if (node->player == maximizer) {
        values[0] = newNode->alpha;
    } 
    if (opponent(node->player) == maximizer) {
        values[0] = newNode->beta;
    }

    /* GPU search the rest of the child nodes */
    int numMoves = moves[0];
    int *dev_moves;
    int *dev_board;
    int *dev_values;
    int *tmoves = (int *)malloc(numMoves * sizeof(int));
    for (int i = 1; i < moves[0]; i++) {
        tmoves[i] = moves[i];
    }

    cudaMalloc((void **) &dev_moves, numMoves * sizeof(int));
    cudaMalloc((void **) &dev_board, BOARDSIZE * sizeof(int));
    cudaMalloc((void **) &dev_values, numMoves * sizeof(int));

    cudaMemcpy(dev_board, &(node->board), BOARDSIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_moves, tmoves, numMoves * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(dev_values, 0, (numMoves) * sizeof(int));

    // call kernel to search the rest of the children in parallel
    cudaMinMaxKernel(dev_moves, dev_board, dev_values, ntm, maximizer, 
        node->alpha, node->beta, numMoves, ply);

    // copy remaining child values into host array
    cudaMemcpy(values, dev_values, numMoves * sizeof(int), cudaMemcpyDeviceToHost);

    // find the best move
    int index = 1;
    if (node->player == maximizer) {
        int best =  WIN+1;
        for (int i = 1; i <= numMoves; i++) {
            if (values[i] < best) {
                best = values[i];
                index = i;
            }
        }
        node->beta = best;
    } else {
        int best = LOSS - 1;
        for (int i = 1; i <= numMoves; i++) {
            if (values[i] > best) {
                best = values[i];
                index = i;
            }
        }
        node->alpha = best;
    }
    // printf("%d\n", moves[index]);

    cudaFree(dev_values);
    cudaFree(dev_board);
    cudaFree(dev_moves);

    return moves[index];
}
    
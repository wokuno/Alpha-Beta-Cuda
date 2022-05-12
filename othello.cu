#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include "othello_kernel.cu"
#include "othello_shared.cu"
#include "alpha_beta.cu"

int counter = 1;

int maxdiffstrategy(int, int *); 

int human (int player, int * board) {
  int move;
  printf("%c to move:", nameof(player)); scanf("%d", &move);
  return move;
}

int randomstrategy(int player, int * board) {
  int r, * moves;
  moves = legalmoves(player, board);
  r = moves[(rand() % moves[0]) + 1];
  free(moves);
  return(r);
}

/* minmax is called to do a "ply" move lookahead, using
evalfn (i.e., the utility function) to evaluate (terminal) boards at 
the end of lookahead. Minmax starts by finding and simulating each legal 
move by player. The move that leads to the best (maximum backed-up score) 
resulting board, is the move (i.e., an integer representing a
board location) that is returned by minmax.
The score of each board (resulting from each move)
is determined by the function diffeval if no player can
move from the resulting board (i.e., game over), by function 
maxchoice if only player can move from the resulting board,
or by function minchoice if the opponent can move from the 
resulting board. 
Importantly, minmax assumes that ply >= 1.
You are to modify minmax so that it exploits alphabeta pruning,
and so that it randomly selects amongst the best moves for player.
*/

int minmax (int player, int * board, int ply) {
  int i, max, ntm, newscore, bestmove, * moves, * newboard;
  int maxchoice (int, int *, int); 
  int minchoice (int, int *, int); 
  moves = legalmoves(player, board); /* get all legal moves for player */
  max = LOSS - 1;  /* any legal move will exceed this score */
  for (i=1; i <= moves[0]; i++) {
    newboard = copyboard(board);
    makemove(moves[i], player, newboard);
    ntm = nexttoplay(newboard, player, 0);
    if (ntm == 0) {  /* game over, so determine winner */
         newscore = diffeval(player, newboard);
         if (newscore > 0) newscore = WIN; /* a win for player */
         if (newscore < 0) newscore = LOSS; /* a win for opp */
    }
    if (ntm == player)   /* opponent cannot move */
       newscore = maxchoice(player, newboard, ply-1);
    if (ntm == opponent(player))
       newscore = minchoice(player, newboard, ply-1);
    if (newscore > max) {
        max = newscore;
        bestmove = moves[i];  /* a better move found */
    }
    free(newboard);
  }
  free(moves);
  return(bestmove);
}

int cudaminmax (int player, int * board, int ply) {
  
  Node *n = new Node;
  n->player = player;
  n->board = board;
  n->parent = NULL;
  
  int bestmove = search(n, player, ply);

  return(bestmove);
}

/* If ply = 0, then maxchoice should return diffeval(player, board), 
else the legal moves that can be made by player from board should
be simulated. maxchoice should return the MAXIMUM board score
from among the possibilities. The backed-up score of each 
board (resulting from each player move) is determined 
by function maxchoice if only player can move from the resulting board,
by function minchoice if the opponent can move from the resulting board,
is WIN if a win for player, a LOSS if a win for opponent, and
a 0 if a draw. 
If two or more boards tie for the maximum backed score, then return
the move that appears first (lowest location) in the moves
array leading to a maximum-score board.
*/

int maxchoice (int player, int * board, int ply) {
  int i, max, ntm, newscore, * moves, * newboard;
  int minchoice (int, int *, int); 
  if (ply == 0) return(diffeval (player, board));
  moves = legalmoves(player, board);
  max = LOSS - 1;
  for (i=1; i <= moves[0]; i++) {
    newboard = copyboard(board);
    makemove(moves[i], player, newboard);
    ntm = nexttoplay(newboard, player, 0);
    if (ntm == 0) {
         newscore = diffeval(player, newboard);
         if (newscore > 0) newscore = WIN;
         if (newscore < 0) newscore = LOSS;
    }
    if (ntm == player) 
       newscore = maxchoice(player, newboard, ply-1);
    if (ntm == opponent(player))
       newscore = minchoice(player, newboard, ply-1);
    if (newscore > max) max = newscore;
    free(newboard);
  }
  free(moves);
  return(max);
}

/* If ply = 0, then minchoice should return the diffeval(player, board), 
else the legal moves that can be made by player's opponent from board should
be simulated. minchoice should return the MINIMUM backed up board score
from among the possibilities. The backed up score of each board 
(resulting from each opponent move) is determined by function maxchoice
if player can move from the resulting board,
by function minchoice if only the opponent can move
from the resulting board, is WIN if a win for player, 
a LOSS if a win for opponent, and a 0 if a draw. 
If two or more BOARDS tie for the minimum score, then return
the move that appears first (lowest location) in the moves
array leading to a minimum-score board.
Advanced: DO NOT worry about this, 
but note that minchoice and maxchoice could be combined
easily into a single function, finding the board
with minimum score, s, is equivalent to finding the board
with maximum -1 * s. One would have to add an additional
parameter to decide whether to multiply by a -1 factor or
not in computing a board score in this combined function.
With a bit more work, one could combine all three
functions, minmax, maxchoice, and minchoice, into a single
function.
*/

int minchoice (int player, int * board, int ply) {
  int i, min, ntm, newscore, * moves, * newboard;
  if (ply == 0) return(diffeval (player, board));
  moves = legalmoves(opponent(player), board);
  min = WIN+1;
  for (i=1; i <= moves[0]; i++) {
    newboard = copyboard(board);
    makemove(moves[i], opponent(player), newboard);
    ntm = nexttoplay(newboard, opponent(player), 0);
    if (ntm == 0) {
         newscore = diffeval(player, newboard);
         if (newscore > 0) newscore = WIN;
         if (newscore < 0) newscore = LOSS;
    }
    if (ntm == player) 
       newscore = maxchoice(player, newboard, ply-1);
    if (ntm == opponent(player))
       newscore = minchoice(player, newboard, ply-1);
    if (newscore < min) min = newscore;
    free(newboard);
  }
  free(moves);
  return(min);
}

int maxdiffstrategy(int player, int * board) { /* 1 ply lookahead */
  return(minmax(player, board, counter));   /* diffeval as utility fn */
}

int cudamaxdiffstrategy(int player, int * board) { /* 1 ply lookahead */
  return(cudaminmax(player, board, counter));   /* diffeval as utility fn */
}

void getmove (int (* strategy) (int, int *), int player, int * board, 
              int printflag) {
  int move;
  if (printflag) printboard(board);
  move = (* strategy)(player, board);
  if (legalp(move, player, board)) {
     if (printflag) printf("%c moves to %d\n", nameof(player), move);
     makemove(move, player, board);
  }
  else {
     printf("Illegal move %d\n", move);
     getmove(strategy, player, board, printflag);
  }
}

int * randomboard (void) {
  int player, oldplayer, i, * board;
  board = initialboard();
  player = BLACK;
  i=1;
  do {
    if (player == BLACK) getmove(randomstrategy, BLACK, board, 0);
    else getmove(randomstrategy, WHITE, board, 0);
    oldplayer = player;
    player = nexttoplay(board, player, 0);
    if (oldplayer == player) {
       free(board);
       return(randomboard());
    }
    i++;
  }
  while (player != 0 && i<=8);
  if (player==0) {
     free(board);
     return(randomboard());
  }
  else return(board);
}

void othello (int algo, bool disp) {
  int * board;
  int player;
  board = initialboard();
  player = BLACK;
  if (algo == 0){
    do {
      if (player == BLACK) getmove(maxdiffstrategy, BLACK, board, disp);
      else getmove(maxdiffstrategy, WHITE, board, disp);
      player = nexttoplay(board, player, disp);
    }
    while (player != 0);
  }
  if (algo == 1) {
      do {
        if (player == BLACK) getmove(maxdiffstrategy, BLACK, board, disp);
        else getmove(cudamaxdiffstrategy, WHITE, board, disp);
        player = nexttoplay(board, player, disp);
      }
      while (player != 0);
  }
  if (algo == 2) {
      do {
        if (player == BLACK) getmove(cudamaxdiffstrategy, BLACK, board, disp);
        else getmove(cudamaxdiffstrategy, WHITE, board, disp);
        player = nexttoplay(board, player, disp);
      }
      while (player != 0);
  }

  printf("[%c=%d %c=%d]\n", 
  nameof(BLACK), count(BLACK, board), nameof(WHITE), count(WHITE, board));
  
  // printboard(board);
}

int main (int argc, char** argv) {
  for (int i=1; i < 7; i++) {
    counter = i;
    for(int j=0; j < 3; j++) {
      if (j == 0) {
        printf("CPU-CPU Depth: %d \n", counter);
      } else if (j == 1) {
        printf("CPU-GPU Depth: %d \n", counter);
      } else if (j == 2) {
        printf("GPU-GPU Depth: %d \n", counter);
      }

      struct timeval start, end;
 
      gettimeofday(&start, NULL);

      othello(j, 0);
      
      gettimeofday(&end, NULL);

      long seconds = (end.tv_sec - start.tv_sec);
      long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
      float millis = ((float) micros) / 1000;

      printf("The elapsed time is %.2f millis\n", millis);
    }
  }
  return 0;
}
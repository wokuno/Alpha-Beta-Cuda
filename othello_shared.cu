#include "othello_shared.h"

/* This code is a translation of Norvig's (1992) LISP implementation, though it deviates from it in several respects. In theory you might be able to track down this code and turn much of it in to satisfy the project requirements, but consistent with the honor code (and ethics generally), you would declare this use and would receive a MUCH reduced grade.
*/

char nameof (int piece) {
  static char piecenames[5] = ".bw?";
  return(piecenames[piece]);
}

int opponent (int player) {
  switch (player) {
  case 1: return 2; 
  case 2: return 1;
  default: printf("illegal player\n"); return 0;
  }
}

int * copyboard (int * board) {
  int i, * newboard;
  newboard = (int *)malloc(BOARDSIZE * sizeof(int));
  for (i=0; i<BOARDSIZE; i++) newboard[i] = board[i];
  return newboard;
}

int * initialboard (void) {
  int i, * board;
  board = (int *)malloc(BOARDSIZE * sizeof(int));
  for (i = 0; i<=9; i++) board[i]=OUTER;
  for (i = 10; i<=89; i++) {
     if (i%10 >= 1 && i%10 <= 8) board[i]=EMPTY; else board[i]=OUTER;
  }
  for (i = 90; i<=99; i++) board[i]=OUTER;
  board[44]=WHITE; board[45]=BLACK; board[54]=BLACK; board[55]=WHITE;
  return board;
}

int count (int player, int * board) {
  int i, cnt;
  cnt=0;
  for (i=1; i<=88; i++)
    if (board[i] == player) cnt++;
  return cnt;
}

void printboard (int * board) {
  int row, col;
  printf("    1 2 3 4 5 6 7 8 [%c=%d %c=%d]\n", 
      nameof(BLACK), count(BLACK, board), nameof(WHITE), count(WHITE, board));
  for (row=1; row<=8; row++) {
    printf("%d  ", 10*row);
    for (col=1; col<=8; col++)
      printf("%c ", nameof(board[col + (10 * row)]));
    printf("\n");
  }    
}

int validp (int move) {
  if ((move >= 11) && (move <= 88) && (move%10 >= 1) && (move%10 <= 8))
     return 1;
  else return 0;
}


/* findbracketingpiece is called from wouldflip (below). 
   findbracketingpiece starts from a *square* that is occupied
   by a *player*'s opponent, moves in a direction, *dir*, past
   all opponent pieces, until a square occupied by the *player*
   is found. If a square occupied by *player* is not found before
   hitting an EMPTY or OUTER square, then 0 is returned (i.e., no
   bracketing piece found). For example, if the current board is
         1 2 3 4 5 6 7 8   
     10  . . . . . . . . 
     20  . . . . . . . . 
     30  . . . . . . . . 
     40  . . b b b . . . 
     50  . . w w w b . . 
     60  . . . . . w . . 
     70  . . . . . . . . 
     80  . . . . . . . . 
   then findbracketingpiece(66, BLACK, board, -11) will return 44
        findbracketingpiece(53, BLACK, board, 1) will return 56
        findbracketingpiece(55, BLACK, board, -9) will return 0
*/


int findbracketingpiece(int square, int player, int * board, int dir) {
  while (board[square] == opponent(player)) square = square + dir;
  if (board[square] == player) return square;
  else return 0;
}

int wouldflip (int move, int player, int * board, int dir) {
  int c;
  c = move + dir;
  if (board[c] == opponent(player))
     return findbracketingpiece(c+dir, player, board, dir);
  else return 0;
}

int legalp (int move, int player, int * board) {
  int i;
  if (!validp(move)) return 0;
  if (board[move]==EMPTY) {
    i=0;
    while (i<=7 && !wouldflip(move, player, board, ALLDIRECTIONS[i])) i++;
    if (i==8) return 0; else return 1;
  }   
  else return 0;
}

void makeflips (int move, int player, int * board, int dir) {
  int bracketer, c;
  bracketer = wouldflip(move, player, board, dir);
  if (bracketer) {
     c = move + dir;
     do {
         board[c] = player;
         c = c + dir;
        } while (c != bracketer);
  }
}

void makemove (int move, int player, int * board) {
  int i;
  board[move] = player;
  for (i=0; i<=7; i++) makeflips(move, player, board, ALLDIRECTIONS[i]);
}

int anylegalmove (int player, int * board) {
  int move;
  move = 11;
  while (move <= 88 && !legalp(move, player, board)) move++;
  if (move <= 88) return 1; else return 0;
}

int nexttoplay (int * board, int previousplayer, int printflag) {
  int opp;
  opp = opponent(previousplayer);
  if (anylegalmove(opp, board)) return opp;
  if (anylegalmove(previousplayer, board)) {
    if (printflag) printf("%c has no moves and must pass.\n", nameof(opp));
    return previousplayer;
  }
  return 0;
}

int cudanexttoplay (int * board, int previousplayer, int printflag) {
  int opp;
  opp = opponent(previousplayer);
  if (anylegalmove(opp, board)) return opp;
  if (anylegalmove(previousplayer, board)) {
    if (printflag) printf("%c has no moves and must pass.\n", nameof(opp));
    return previousplayer;
  }
  return opp;
}


int * legalmoves (int player, int * board) {
  int move, i, * moves;
  moves = (int *)malloc(65 * sizeof(int));
  moves[0] = 0;
  i = 0;
  for (move=11; move<=88; move++) 
    if (legalp(move, player, board)) {
      i++;
      moves[i]=move;
    }
  moves[0]=i;
  return moves;
}

int diffeval (int player, int * board) { /* utility is measured */
    int i, ocnt, pcnt, opp;                /* by the difference in */
    pcnt=0; ocnt = 0;                      /* number of pieces */
    opp = opponent(player); 
    for (i=1; i<=88; i++) {
      if (board[i]==player) pcnt++;
      if (board[i]==opp) ocnt++;
    }
    return (pcnt-ocnt);
  }
#include "magma_v2.h"

void GHDEL(int dim, double *d_A, int cols, double *d_B, int LD, magma_queue_t queue)
{
  
  for(int i = 0; i< dim; i++)
  {
// row elimination
 magmablas_dgemv(MagmaTrans,i,cols,-1,d_B,LD,d_A + i,LD,1,d_B + i ,LD,queue);
  //scaling
 magma_dtrsm(MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, 1, cols, 1, d_A + i * LD + i, LD, d_B + i, LD,queue);
 
  //column elimination
   magma_dger(i,cols,-1,d_A + i * LD,1,d_B + i,LD,d_B,LD,queue);         
  
 }

}




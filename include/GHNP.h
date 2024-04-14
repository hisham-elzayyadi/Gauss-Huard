#include "magma_v2.h"

void GHNP(int rows, int cols, int LDB, double *B, magma_queue_t queue)
{
 
  for(int i = 0; i< rows; i++)
  {
// row elimination
  magmablas_dgemv(MagmaTrans,i,cols  - i,-1,B+ i * LDB,LDB,B + i,LDB,1,B + i * LDB + i,LDB,queue);
    
  //scaling
   magma_dtrsm(MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, 1,
                  cols -1 - i, 1, B + i * LDB + i, LDB, B + (i + 1) * LDB + i, LDB, queue);
             

   // if(i < rows - 1)	
  //column elimination
    magma_dger(i,cols - i - 1,-1,B+ i * LDB,1,B + (i + 1) * LDB + i,LDB,B + (i + 1) * LDB,LDB,queue); 
     
  }

  
}





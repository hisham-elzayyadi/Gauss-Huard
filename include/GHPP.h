#include "magma_v2.h"



void GHPP(int rows,int cols, double *B ,double *d_A,int LDB, magma_queue_t queue)
{

    int IndexOfMax;
    
    

       //start Block elimination
        for (int i = 0; i < rows ; i++)
        {
            //row elimination
            magma_dgemm(MagmaNoTrans, MagmaNoTrans, /*M*/ 1, /*N*/ cols - i, /*k*/ i, /*alpha*/ -1,
                        /*A*/ B + i, /*LDA*/ LDB, /* B*/ B + i * LDB, /*LDB*/ LDB, /*beta*/ 1, /*c*/ B + i * LDB + i, /*LDC*/ LDB, queue);

            //pivoting 
            IndexOfMax = magma_idamax(cols - 1 - i, B + i * LDB + i, LDB, queue);
            magma_dswap(LDB, d_A + i * LDB, 1, d_A + (IndexOfMax + i - 1) * LDB, 1, queue);
            //scaling
               magma_dtrsm(MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, 1,
                    cols -1 - i, 1, B + i * LDB + i, LDB, B + (i + 1) * LDB + i, LDB, queue);

            //column elimination
            magma_dgemm(MagmaNoTrans, MagmaNoTrans, /*M*/ i, /*N*/ cols - i , /*k*/ 1, /*alpha*/ -1, /*A*/ B+ i * LDB,
                    /*LDA*/ LDB, /* B*/ B + (i + 1) * LDB + i, /*LDB*/ LDB, /*beta*/ 1, /*c*/ B + (i + 1) * LDB, /*LDC*/ LDB, queue);
        }

        
    }
    




#include <cblas.h>

void GHNPCPU(int dim, double *B)
{


       for(int i = 0; i< dim; i++)
  {
// row elimination
cblas_dgemv(CblasColMajor, CblasTrans, i, dim - i, -1, B + i * dim, dim, B + i, dim, 1, B + i * dim + i, dim);

// scaling
cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 1, dim -1 - i, 1, B + i * dim + i, dim, B + (i + 1) * dim + i, dim);

// column elimination
if(i < dim - 1)  
    cblas_dger(CblasColMajor, i, dim - i - 1, -1, B + i * dim, 1, B + (i + 1) * dim + i, dim, B + (i + 1) * dim, dim);
        
    }
    
}




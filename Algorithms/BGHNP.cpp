#include <bits/stdc++.h>
#include <cmath>
#include <fstream>
#include "magma_v2.h"
#include <cuda_runtime.h>


using namespace std;

int main(int argc, char *argv[])
{

    //enter the size of the matrix A
    int n = atoi(argv[1]);
    //get the block size
    cout << "enter block size:";
    int BlockSize;
    cin >> BlockSize;

  
    float milliseconds = 0;
    float temp = 0;
    //allocate space for matrix A and solution vector x
    double *A = new double[n * (n + 1)]();
    double *b = new double[n]();
    double *x = new double[n]();

    int info;
    //read the file
    ifstream File;
    File.open("matrix.txt");
    int i = 0, j = 0;
    while (File >> *(A + i * n + j))
    {
        i++;
        if (i == n)
        {
            j++;
            i = 0;
        }
    }
    File.close();

    //put the right hand side as the summation of all columns
    for (j = 0; j < n; j++)
    {

        for (i = 0; i < n; i++)
        {
            *(A + n * n + j) = *(A + n * n + j) + *(A + i * n + j);
        }
    }

    magma_init();
     magma_int_t dev = 0;

    magma_queue_t queue1 = NULL;
    magma_queue_create(dev, &queue1);

    magma_queue_t queue2 = NULL;
    magma_queue_create(dev, &queue2);



    
    
    double *d_A;
    magma_dmalloc(&d_A, n * (n + 1));
    
    double *B;
    magma_dmalloc_pinned(&B, BlockSize * BlockSize);

    for (int q = 0; q < 1; q++)
    {
    
    magma_setmatrix(n, n + 1, sizeof(double), A, n, d_A, n, queue1);
        
        //record time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
       
        
        //magma_dgerbt_gpu(MagmaTrue, n, 1, d_A, n, d_A + n * n, n, U, V, &info);
      

        //start Block elimination
        for (i =0; i < n; i += BlockSize)
        {
            //row elimination
            magma_dgemm(MagmaNoTrans, MagmaNoTrans, /*M*/ BlockSize, /*N*/ BlockSize, /*k*/ i, /*alpha*/ -1,
                        /*A*/ d_A + i, /*LDA*/ n, /* B*/ d_A + i * n, /*LDB*/ n, /*beta*/ 1, /*c*/ d_A + i * n + i, /*LDC*/ n, queue1);
            magma_queue_sync(queue1);
            //row elimination
            magma_dgemm(MagmaNoTrans, MagmaNoTrans, /*M*/ BlockSize, /*N*/ n + 1 - i - BlockSize, /*k*/ i, /*alpha*/ -1,
                        /*A*/ d_A + i, /*LDA*/ n, /* B*/ d_A + (i + BlockSize) * n, /*LDB*/ n, /*beta*/ 1, /*c*/ d_A + (i + BlockSize) * n + i, /*LDC*/ n, queue1);
           
             //LU factorization for B panel
            magma_getmatrix(BlockSize, BlockSize, sizeof(double), d_A + i * n + i, n, B, BlockSize, queue2);
            magma_dgetrf_nopiv(BlockSize, BlockSize, B, BlockSize, &info);
            magma_setmatrix(BlockSize, BlockSize, sizeof(double), B, BlockSize, d_A + i * n + i, n, queue2);
            
           
            
            magma_dtrsm(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit, BlockSize,
                    n + 1 - BlockSize - i, 1, d_A + i * n + i, n, d_A + (i + BlockSize) * n + i, n, queue1);
            magma_dtrsm(MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, BlockSize,
                    n + 1 - BlockSize - i, 1, d_A + i * n + i, n, d_A + (i + BlockSize) * n + i, n, queue1);
            //column elimination
            magma_dgemm(MagmaNoTrans, MagmaNoTrans, /*M*/ i, /*N*/ n + 1 - i - BlockSize, /*k*/ BlockSize, /*alpha*/ -1, /*A*/ d_A + i * n,
                        /*LDA*/ n, /* B*/ d_A + (i + BlockSize) * n + i, /*LDB*/ n, /*beta*/ 1, /*c*/ d_A + (i + BlockSize) * n, /*LDC*/ n, queue1);
        }
         
        
        //  magma_dsetmatrix(2,n, V, 2 ,d_V, 2,queue1);
         // magmablas_dprbt_mv(n, d_V,d_A+n*n,queue1);
          
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp, start, stop);
        milliseconds = milliseconds + temp;
       // cout << temp << endl;
    }
   // magma_dprint_gpu(n, 1, d_A + n * n, n, queue1);
   // milliseconds = milliseconds / 100;
   // cout << "the average time is: " << milliseconds << endl;


    /* Componentwise backward error*/
    //residual calculation
    double *r = new double[n]();
    double *d_r;
    magma_dmalloc(&d_r, n);

    magma_setvector(n, sizeof(double), A + n * n, 1, d_r, 1, queue1);
    magma_setmatrix(n, n, sizeof(double), A, n, d_A, n, queue1);
    /* calculate the infinity norm of A*/
    double *dwork;
    magma_dmalloc(&dwork, n);
    double infinityNorm = magmablas_dlange(MagmaInfNorm, n, n, d_A, n, dwork, n, queue1);
    /*-------------------------------------------------------------------*/
    magma_dgemv(MagmaNoTrans, n, n, 1, d_A, n, d_A + n * n, 1, -1, d_r, 1, queue1);

    magma_getvector(n, sizeof(double), d_r, 1, r, 1, queue1);
    for (i = 0; i < n; i++)
    {
        r[i] = abs(r[i]);
    }

    //absolute value of A
    for (j = 0; j < n; j++)
    {

        for (i = 0; i < n; i++)
        {
            *(A + i * n + j) = abs(*(A + i * n + j));
        }
    }
    magma_setmatrix(n, n, sizeof(double), A, n, d_A, n, queue1);
    //absolute value of x
    magma_getvector(n, sizeof(double), d_A + n * n, 1, x, 1, queue1);
    for (i = 0; i < n; i++)
    {
        x[i] = abs(x[i]);
    }
    magma_setvector(n, sizeof(double), x, 1, d_A + n * n, 1, queue1);
    //absolute value of b
    double *d_b;
    magma_dmalloc(&d_b, n);
    for (i = 0; i < n; i++)
    {
        *(A + n * n + i) = abs(*(A + n * n + i));
    }
    magma_setvector(n, sizeof(double), A + n * n, 1, d_b, 1, queue1);
    //|A|*|x|+|b|
    magma_dgemv(MagmaNoTrans, n, n, 1, d_A, n, d_A + n * n, 1, 1, d_b, 1, queue1);
    magma_getvector(n, sizeof(double), d_b, 1, b, 1, queue1);

    // max |Ax-b|/(|A||x|+|b|)
    for (i = 0; i < n; i++)
    {
        b[i] = r[i] / b[i];
    }

    /* Forward Error */
    magma_getvector(n, sizeof(double), d_A + n * n, 1, x, 1, queue1);
    for(i=0;i<n;i++)
{
  x[i]=x[i]-1;
  x[i] = abs(x[i]);
}

    cout<<"Forward Error is: " << *max_element(x, x + n - 1) << endl;
    cout << "Componentwise Backward error is " << *max_element(b, b + n - 1) << endl;
    /* normwise (infinity norm) backward error*/
    cout << "normwise backward error is: " << *max_element(r, r + n - 1) / (infinityNorm * *max_element(x, x + n - 1) + *max_element(A + n * n, A + n * n + n - 1));

    delete A;
    delete x;
    delete b;
    delete r;
    magma_free(d_A);
    magma_free(d_r);
    magma_free(d_b);
    
    magma_free_pinned(B);
    
    magma_queue_destroy(queue1);
    magma_queue_destroy(queue2);
    magma_finalize();
}








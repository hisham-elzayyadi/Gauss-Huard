#include <bits/stdc++.h>
#include <cmath>
#include <fstream>
#include "magma_v2.h"
#include <cuda_runtime.h>
#include "GHDEL.h"
#include "GHNP.h"
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
    double *U = new double[2 * n]();
    double *V = new double[2 * n]();

    int info;
    //read the file
    ifstream File;
    File.open("matrix20.txt");
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
    for (int j = 0; j < n; j++)
    {

        for (int i = 0; i < n; i++)
        {
            *(A + n * n + j) = *(A + n * n + j) + *(A + i * n + j);
        }
    }

    magma_init();
    magma_queue_t queue1 = NULL;
    magma_queue_t queue2 = NULL;
    magma_int_t dev = 0;
    magma_queue_create(dev, &queue1);
  magma_queue_create(dev, &queue2);
   

    double *d_V;
    magma_dmalloc(&d_V, n * 2);
    //generating random diagonal numbers and store them in U and V
    double *d_A;
    magma_dmalloc(&d_A, n * (n + 1));
    double *B;
    magma_dmalloc_pinned(&B, BlockSize * BlockSize);
    for (int j = 0; j < 30; j++)
    {
      magma_setmatrix(n, n + 1, sizeof(double), A, n, d_A, n, queue1);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
       
        
        magma_dgerbt_gpu(MagmaTrue, n, 1, d_A, n, d_A + n * n, n, U, V, &info);

        for(int i = 0; i< n; i+= BlockSize)
  {
              magma_dgemm(MagmaNoTrans, MagmaNoTrans, /*M*/ BlockSize, /*N*/ n + 1  - i, /*k*/ i, /*alpha*/ -1,
                    /*A*/ d_A + i, /*LDA*/ n, /* B*/ d_A+ i * n, /*LDB*/ n, /*beta*/ 1, /*c*/ d_A + i * n + i, /*LDC*/ n, queue1);
    //Block diagonalization
   
           GHNP(BlockSize,BlockSize,n, d_A + i * n + i, queue1);
      
      // A12 update
        GHDEL(BlockSize,d_A + i * n + i, n + 1 - BlockSize -i,d_A + (i+BlockSize) * n + i , n, queue1);
      
    //column elimination
       magma_dgemm(MagmaNoTrans, MagmaNoTrans, /*M*/ i, /*N*/ n + 1 - i - BlockSize, /*k*/ BlockSize, /*alpha*/ -1, /*A*/ d_A + i * n,
                        /*LDA*/ n, /* B*/ d_A + (i + BlockSize) * n + i, /*LDB*/ n, /*beta*/ 1, /*c*/ d_A + (i + BlockSize) * n, /*LDC*/ n, queue1);
       }
      
         magma_dsetmatrix(2,n, V, 2 ,d_V, 2,queue1);
          magmablas_dprbt_mv(n, d_V,d_A+n*n,queue1);
          
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp, start, stop);
        milliseconds = milliseconds + temp;
        cout << temp << endl;
    }
   // magma_dprint_gpu(n, 1, d_A + n * n, n, queue1);
    milliseconds = milliseconds / 30;
    cout << "the average time is: " << milliseconds << endl;

    delete A;
    delete U;
    delete V;

    magma_free(d_A);
    magma_free(d_V);
   magma_free_pinned(B); 
    magma_queue_destroy(queue1);
    magma_queue_destroy(queue2);
    magma_finalize();
}






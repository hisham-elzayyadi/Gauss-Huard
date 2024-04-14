#include <iostream>
#include <fstream>
#include "magma_v2.h"
#include <cuda_runtime.h>


using namespace std;





int main(int argc, char *argv[])
{

    //enter the size of the matrix A
    int n = atoi(argv[1]);

    //allocate space for matrix A and solution vector x
    double *A = new double[n * (n+1)]();
    double *U = new double[2 * n]();
    double *V = new double[2 * n]();
    
    int info;
    float milliseconds = 0;
    float temp = 0;
    
      //read the file
    ifstream File;
    File.open("matrix5.txt");
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
    for ( j = 0; j < n; j++)
    {
        
        for ( i = 0; i < n; i++)
        {
             *(A + n * n + j) = *(A + n * n + j) + *(A + i * n + j);
        }
    }
   magma_init();
   magma_queue_t queue1 = NULL;
    magma_int_t dev = 0;
    magma_queue_create(dev, &queue1);

    double *d_A;
    magma_dmalloc(&d_A, n * (n + 1));
   
    double *d_V;
    magma_dmalloc(&d_V, 2 * n);

    
     for(int i=0; i<100; i++)
     {
  
        magma_setmatrix(n, n + 1, sizeof(double), A, n, d_A, n, queue1);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

      
     
     magma_dgerbt_gpu(MagmaTrue, n, 1, d_A, n, d_A + n * n, n, U, V, &info);
     magma_dgesv_nopiv_gpu(n,1,d_A,n,d_A+n*n,n,&info);
      magma_dsetmatrix(2,n, V, 2 ,d_V, 2,queue1);
      magmablas_dprbt_mv(n, d_V,d_A+n*n,queue1);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp, start, stop);
        milliseconds = milliseconds + temp;
         cout << temp << endl;
     }
    milliseconds = milliseconds / 100;

    //magma_dprint_gpu(n,1,d_A + n*n,n,queue1);    
    cout << "the average time is :" << milliseconds << endl;
   
    delete A;
    delete U;
    delete V;
    magma_free(d_A);
    magma_free(d_V);
    magma_finalize();

}








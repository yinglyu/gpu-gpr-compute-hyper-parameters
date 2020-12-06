#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

__device__ int count;

__device__ void sum(double *partial_sum, int dummy) {
    if(threadIdx.x == 0) {
        count = dummy;
        if(count %2 != 0) {
            count++;
            partial_sum[count-1] = 0;
        }
    }
    __syncthreads();
    for(int i = count/2; i > 0; i = i/2) {
        if(threadIdx.x < i)
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + i];
        __syncthreads();
        if(threadIdx.x == 0) {
            if(i%2 != 0 && i != 1) {
                partial_sum[0] += partial_sum[--i];
            }
        }
        __syncthreads();
    }
    __syncthreads();
    return;
}

void init_grid_points(double * x, double * y, int m)
{
    double h = (double)1/(m + 1);
    for (int i = 0; i < m; i ++)
    {
        for (int j = 0; j < m; j ++)
        {
            x[i*m+j] = (i + 1)*h;
            y[i*m+j] = (j + 1)*h;
        }
    }
}

void init_observed_data_vector(double * f, double * x, double * y, int m)
{
    double l[2] = {(double)2/m, (double)2/m};
    int n = m * m;
    //kernel(f, x, y, l, n);
    for (int i = 0; i < n; i ++)
    {
        f[i] = 0.02 * ((double)rand() / (double)RAND_MAX - 0.5);
        double d = pow((x[i] - 0.25)/l[0], 2) + pow((y[i] - 0.25)/l[1],2);
        f[i] += 1.0/sqrt(2.0*M_PI) *exp(-d/2);
        f[i] += x[i] * 0.2 + y[i] * 0.1;
    } 
}

void randperm(int * r, int n){
    for(int i = 0; i < n; i ++){
        r[i] = i;
    }
    for (int i = n - 1; i >= 0; i --){
        int j = rand() % (i + 1);
        int temp = r[i];
        r[i] = r[j];
        r[j] = temp;
    }
}

void init_data_set_indices(int * itest, int * itrain, int ntest, int ntrain){
    int n = ntest + ntrain;
    int * r = (int *) malloc(n * sizeof(int));
    randperm(r, n);
    for (int i = 0; i < ntest; i ++){
        itest[i] = r[i];
    }
    for (int i = 0; i < ntrain; i ++){
        itrain[i] = r[ntest + i];
    }
    free(r);
}

void compute_A(double * A, double t, int n)//tI + K
{
    //Compute A = tI+K
    for (int i = 0; i < n; i ++)
    {
        A[i*n + i] += t;
    }
}

void compute_k(double * k, double * x, double * y, double * rstar, int n)
{
    int i; 
    double d;
    for (i = 0; i < n; i ++)
    {
        d = pow(rstar[0]-x[i], 2) + pow(rstar[1]-y[i], 2);
        k[i] = exp(-d);
    }
}

void compute_LU_factors(double * A, int n)
{
    int k, i, j;
    double m;

    for (k = 0; k < n - 1; k ++)
    {
        for (i = k + 1; i < n; i ++)
        {
            m = A[i*n + k] / A[k*n + k];
            for (j = k + 1; j < n; j ++)
            {
                A[i*n + j] = A[i*n + j] - m * A[k*n + j];
            }
            A[i*n + k] = m;
        }
    }
}

__global__ void compute_LU_factors(int N, double * A, int n)
{
    int k, i, j;
    int m;
    for (k = 0; k < n - 1; k ++)
    {
        for (i = k + 1 + threadIdx.x; i < n; i += N)
        {
            A[i*n + k] = A[i*n + k] / A[k*n + k];
        }
        __syncthreads();
        for (m = threadIdx.x; m < (n - k - 1)*(n - k - 1); m += N )
        {
        	i = k + 1 + m / (n - k - 1);
        	j = k + 1 + m % (n - k - 1);
        	A[i*n + j] -=  A[i*n + k] * A[k*n + j];
        }
        __syncthreads();
    }
    return;
}

void solve_triangular_systems(double * z, double * A, double * f, int * itrain, int n)
{
    int i, j;
    double m; 
    //Solve Az = f by LUz = f
    //1. Solve Ly = f for y
    for (i = 0; i < n; i ++)
    {
        m = 0;
        for (j = 0; j < i; j ++)
        {
            m += A[i*n + j] * z[j];
        }

        z[i] = f[itrain[i]] - m;
    }
    
    //2. Solve Uz = y for z
    for (i = n - 1; i >= 0; i --)
    {
        m = 0;
        for (j = i + 1; j < n; j ++)
        {
            m += A[i*n + j] * z[j];
        }
        z[i] = (z[i]-m)/A[i*n + i];
        
    }
}

__global__ void solve_triangular_systems(int N, double * z, double * A, double * f, int n)
{
    extern __shared__ double partial_sum[];
    int i, j;
    //Solve Az = f by LUz = f
    //1. Solve Ly = f for y
    for (i = 0; i < n; i ++)
    {
        partial_sum[threadIdx.x] = 0;
        for (j = threadIdx.x; j < i; j += N)
        {
            partial_sum[threadIdx.x] += A[i*n + j] * z[j];
        }
        sum (partial_sum, (N<i)?N:i);
        if (threadIdx.x == 0){
            z[i] = f[i] - partial_sum[0];
        }
        __syncthreads();
    }
    __syncthreads();
    
    //2. Solve Uz = y for z
    for (i = n - 1; i >= 0; i --)
    {
        partial_sum[threadIdx.x] = 0;
        for (j = i + 1 + threadIdx.x; j < n; j += N)
        {
            partial_sum[threadIdx.x] += A[i*n + j] * z[j];
        }
        __syncthreads();
        sum(partial_sum, (N < (n-1-i))? N:(n-1-i));
        if(threadIdx.x == 0) {
            z[i] = (z[i]-partial_sum[0])/A[i*n + i];
        }
        __syncthreads();
    }
    return;
}

double compute_fstar(double * k, double * z, int n)
{
    int i;
    double fstar = 0.0;
    // Compute predicted value fstar at rstar: k'*z
    for (i = 0; i < n; i ++)
    {
        fstar += k[i] * z[i];
    }
    
    return fstar;    
}

void compute_ftest(double * ftest, double * k, double * z, int ntrain, int ntest)
{
    // Compute predicted value ftest at itest array: kT*z
    for (int i = 0; i < ntest; i ++)
    {
        ftest[i] = 0;
        for (int j = 0; j < ntrain; j ++){
            ftest[i] += k[i * ntrain + j] * z[j];
        }
    }
}

void compute_kernel(double * K, double * x, double * y, double * l, int n)
{
    for (int i = 0; i < n; i ++)
    {
        for (int j = 0; j < n; j++)
        {
            double d = pow((x[i] - x[j])/l[0], 2) + pow((y[i] - y[j])/l[1],2);
            K[i*n + j] = 1.0/sqrt(2.0*M_PI) * exp(-d/2);
        }
    } 
}

void extract_K(double * K0, double * K, int * i1, int * i2, int n, int n1, int n2){
    for (int i = 0; i < n1; i ++)
    {
        for (int j = 0; j < n2; j++)
        {
            K[i * n1 + j] = K0[i1[i] * n + i2[j]];
        }
    }
}

void print_array(double * array, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%.4f ", array[i]);
    }
    printf("\n");
}

void print_matrix(double * matrix, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.4f ", matrix[i*m + j]);
        }
        printf("\n");
    }
}

void GPR(double * ftest, double * x, double * y, double * f, int * itest, int * itrain, double t, double * l, int n, int ntest)
{
    double * K0;
    double * LU;
    // double * k;
    double * kT;
    double * z;
    
    
    int ntrain = n - ntest;
    K0 = (double *) malloc(n * n * sizeof(double));
    
    //Initialize K for all points (including test and training points)
    compute_kernel(K0, x, y, l, n);
    
    //Extract training set K
    LU = (double *) malloc(ntrain * ntrain * sizeof(double));
    extract_K(K0, LU, itrain, itrain, n, ntrain, ntrain);
    
    compute_A(LU, t, ntrain);//tI + K
    
    compute_LU_factors(LU, ntrain);

    kT = (double *) malloc(ntest * ntrain * sizeof(double));
    extract_K(K0, kT, itest, itrain, n, ntest, ntrain);

    z = (double *) malloc(ntrain * sizeof(double));
    solve_triangular_systems(z, LU, f, itrain, ntrain);
   
    compute_ftest(ftest, kT, z, ntrain, ntest);

    free(K0);
    free(LU);
    free(kT);
    free(z);
}

double compute_MSE(double * f, int * itest, double * ftest, int ntest) // compute the mean square error
{
    double squareError = 0;
    for (int i = 0; i < ntest; i ++){
        squareError += pow(f[itest[i]] - ftest[i], 2);
    }
    return squareError / ntest;
}



int main(int argc, char** argv) 
{

	// Host Data
	double * hGx;  // host grid x-coordinate array
	double * hGy;  // host grid y-coordinate array
	double * hf;// host observed data vector f
    int * hitest; // Indices of test points (randomly chosen)
    int * hitrain; //Indices of training points

	// Grid size m, grid points n, size of test data and training data, 
    int m = 4, n, ntest, ntrain;

    // Coordinate of hyper-parameter l(l1, l2)
    double l[2], bestL[2];

    // predicted value of test
    double * ftest;
    
    // Timing variables
    // cudaEvent_t start, stop; // GPU timing variables

    // Other variables
    // double fstar;
    double Lparam[20];
    double MSE[20][20];
    int size;
    double minMSE = DBL_MAX;

    // Timing initializations
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

	// Check input    
    if (argc > 1){
        m = atoi(argv[1]);
    }else{
        printf("Please indicate grid size m");
        return -1;
    }
    
    // Allocate host coordinate arrays
    n = m * m;
    size = n * sizeof(double);
    hGx = (double *) malloc(size);
    hGy = (double *) malloc(size);
    hf = (double *) malloc(size); 

    ntest = (n + 9) / 10;
    ntrain = n - ntest;
    printf("testing data: %d, training data: %d\n", ntest, ntrain);
    size = sizeof(int);
    hitest = (int *) malloc(ntest * size);
    hitrain = (int *) malloc(ntrain * size);
    size = sizeof(double);
    ftest = (double *) malloc(ntest * size);

    for (int i = 0; i < 20; i++){
        Lparam[i] = (i + 1)  * 0.5/ m;
    }
    
    init_grid_points(hGx, hGy, m);
    
    srand(time(0));
    init_observed_data_vector(hf, hGx, hGy, m);

    init_data_set_indices(hitest, hitrain, ntest, ntrain);
    
    double t = 0.5;// Parameter t
    for (int il1 = 0; il1 < 20; il1 ++){
        l[0] = Lparam[il1];
        for (int il2 = 0; il2 < 20; il2 ++){
            l[1] = Lparam[il2];
            GPR(ftest, hGx, hGy, hf, hitest, hitrain, t, l, n, ntest);
            MSE[il1][il2] = compute_MSE(hf, hitest, ftest, ntest);
            printf("Finished (l1,l2) = %f, %f, mse = %e\n", Lparam[il1], Lparam[il2], MSE[il1][il2]);
            if (MSE[il1][il2] < minMSE){
                bestL[0] = l[0];
                bestL[1] = l[1];
                minMSE = MSE[il1][il2];
            }
            
        }
    }
    printf("Best (l1,l2) = %f, %f, mse = %e\n", bestL[0], bestL[1], minMSE);
    

    free(hGx);
    free(hGy);
    free(hitest);
    free(hitrain);
    free(ftest);
    
    return 0;
}

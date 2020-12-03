#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

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

void init_observed_data_vector(double * f, double * x, double * y, int size)
{
    for (int i = 0; i < size; i++)
    {
        f[i] = 1.0 - pow(x[i] - 0.5, 2) - pow(y[i] - 0.5, 2) + 0.1 * ((double)rand() / (double)RAND_MAX - 0.5);
    }
}

void compute_A( double * A, double * x, double * y, int n)
{
    int i, j;
    double d, t;
    //Initialize K
    for (i = 0; i < n; i ++)
    {
        for (j = 0; j < n; j++)
        {
            d = pow(x[i] - x[j], 2) + pow(y[i] - y[j],2);
            A[i*n + j] = exp(-d);
        }
    }
    //Compute A = tI+K
    t = 0.01;
    for (i = 0; i < n; i ++)
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

void solve_triangular_systems(double * z, double * A, double * f, int n)
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

        z[i] = f[i] - m;
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

void print_array(double * array, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%.2f ", array[i]);
    }
    printf("\n");
}

void print_matrix(double * matrix, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.2f ", matrix[i*m + j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) 
{

	// Host Data
	double * hGx;  // host grid x-coordinate array
	double * hGy;  // host grid y-coordinate array
	double * hA;  // host tI+K
	double * hLU; // host LU factorization of A
	
	double * hf;// host observed data vector f
	double * hk;// host vector k
	double * hz;// host triangular systems solution

	// Device Data
	// double * dGx;  // device grid x-coordinate array
	// double * dGy;  // device grid y-coordinate array
	double * dA;  // device tI+K
	//double * hLU; // device LU factorization of A
	
	double * df;// device observed data vector f
	// double * dk;// device vector k
	double * dz;// device triangular systems solution


	// Grid size m, grid points n
    int m = 4, n;

    // Coordinate of r*
    double * rstar;
    rstar = (double *) malloc(2 * sizeof(double));
    
    // Timing variables
    float LU_time, solver_time, total_time;
    cudaEvent_t start, stop; // GPU timing variables

    // Other variables
    double fstar;
    int size;

    // Timing initializations
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	// Check input    
    if (argc > 3){
        m = atoi(argv[1]);
        rstar[0] = atof(argv[2]);
        rstar[1] = atof(argv[3]);
        printf("r*=(%lf, %lf)\n", rstar[0], rstar[1]);
    }else{
        // cout << "Please indicate grid size and coordinate of r*" << endl;
        printf("Please indicate grid size and coordinate of r*");

        return -1;
    }
    
    // Allocate host coordinate arrays
    n = m * m;
    size = n * sizeof(double);
    hGx = (double *) malloc(size);
    hGy = (double *) malloc(size);
    hf = (double *) malloc(size); 
    hk = (double *) malloc(size);
    hz = (double *) malloc(size);
    size = n * n * sizeof(double);
    hA = (double *) malloc(size);
    hLU = (double *) malloc(size);
    // printf("Allocate host coordinate arrays\n");
    
    init_grid_points(hGx, hGy, m);
    // printf("x and y coordinates of grid points\n");
    // print_array(hGx, n);
    // print_array(hGy, n);
    
    srand(time(0));
    init_observed_data_vector(hf, hGx, hGy, n);
    // printf("observed data vector f\n");
    // print_array(hf, n);
    
    compute_A(hA, hGx, hGy, n);//tI+K
    // printf("compute_A\n");
    // print_matrix(hA, n, n);
    
    compute_k(hk, hGx, hGy, rstar, n);
    // printf("compute_k\n");
    // print_array(hk, n);
    
    // LU_floats = n*(n-1)*(4*n+1);
    // LU_floats /= 6;
    // solver_floats = n*(4+n);

    // Allocate device coordinate arrays
    size = n * sizeof(double);
    cudaMalloc(&df, size);
    cudaMemcpy(df, hf, size, cudaMemcpyHostToDevice);
    cudaMalloc(&dz, size);
    size = n * n * sizeof(double);
    cudaMalloc(&dA, size);
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    printf("GPU version\n");
    int threads = 192;
    printf("Number of threads %d\n", threads);
    cudaEventRecord(start, 0); 
    compute_LU_factors<<<1, threads>>>(threads, dA, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&LU_time, start, stop);
    size = n * n * sizeof(double);
    cudaMemcpy(hLU, dA, size, cudaMemcpyDeviceToHost);

    printf("LU time = %f ms\n", LU_time);
    cudaEventRecord(start, 0); 
    solve_triangular_systems<<<1, threads, threads * sizeof(double)>>>(threads, dz, dA, df, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&solver_time, start, stop);
    size = n * sizeof(double);
    cudaMemcpy(hz, dz, size, cudaMemcpyDeviceToHost);

    printf("Solver time = %f ms\n", solver_time);

    total_time = LU_time + solver_time;
    
    fstar = compute_fstar(hk, hz, n);
    printf("Total time = %f ms, Predicted value = %lf\n", total_time, fstar);


    cudaFree(df);
    cudaFree(dz);
    cudaFree(dA);

    free(hGx);
    free(hGy);
    free(hA);
    free(hLU);
    free(hf);
    free(hk);
    free(hz);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


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
    // for (i = 0; i < n; i ++){
    //     LU[i * n] = A[i * n];
    //     LU[i] = A[i];
    // }

    for (k = 0; k < n - 1; k ++)
    {
        // # pragma omp parallel for shared(A) private(i, j, m) proc_bind(close) 
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
	//double * hLU; // host LU factorization of A
	
	double * hf;// host observed data vector f
	double * hk;// host vector k
	double * hz;// host triangular systems solution

	// Grid size m, grid points n
    int m = 4, n;

    // Coordinate of r*
    double * rstar;
    rstar = (double *) malloc(2 * sizeof(double));
    
    // Timing variables
    float LU_time, solver_time, total_time, LU_floats, solver_floats, LU_FLOPS, solver_FLOPS;
    struct timespec cpu_start, cpu_stop; // CPU timing variables

    // Other variables
    double fstar;
    int size;

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
    //hLU = (double *) malloc(size);
    printf("Allocate host coordinate arrays\n");
    
    init_grid_points(hGx, hGy, m);
    printf("x and y coordinates of grid points\n");
    print_array(hGx, n);
    print_array(hGy, n);
    
    srand(time(0));
    init_observed_data_vector(hf, hGx, hGy, n);
    printf("observed data vector f\n");
    print_array(hf, n);
    
    compute_A(hA, hGx, hGy, n);//tI+K
    printf("compute_A\n");
    print_matrix(hA, n, n);
    
    compute_k(hk, hGx, hGy, rstar, n);
    printf("compute_k\n");
    print_array(hk, n);
    
    double N = (double) n;
    LU_floats = N*(N-1)*(4*N+1)/6;
    solver_floats = N*(4+N);
    clock_gettime(CLOCK_REALTIME, &cpu_start);
    
    compute_LU_factors(hA, n); //LU factorization of A
   clock_gettime(CLOCK_REALTIME, &cpu_stop);
    LU_time = 1000*((cpu_stop.tv_sec-cpu_start.tv_sec) + 0.000000001*(cpu_stop.tv_nsec-cpu_start.tv_nsec));
    LU_FLOPS = LU_floats/LU_time;
    printf("LU factorization of A\n");
    printf("LU_FLOPS = %f\n", LU_FLOPS);
    print_matrix(hA, n, n);
 
    clock_gettime(CLOCK_REALTIME, &cpu_start); 
    solve_triangular_systems(hz, hA, hf, n);
    clock_gettime(CLOCK_REALTIME, &cpu_stop);
    solver_time = 1000*((cpu_stop.tv_sec-cpu_start.tv_sec) + 0.000000001*(cpu_stop.tv_nsec-cpu_start.tv_nsec));
    solver_FLOPS = solver_floats/solver_time;
    printf("solve_triangular_systems\n");
    printf("solver_FLOPS = %f\n", solver_FLOPS);
    print_array(hz, n);
    
    total_time = LU_time + solver_time;
     
    fstar = compute_fstar(hk, hz, n);
    printf("Total time = %lf seconds, Predicted value = %lf\n", total_time, fstar);

    return 0;
}

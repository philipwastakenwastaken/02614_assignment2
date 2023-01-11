/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "alloc3d.h"
#include "print.h"

#ifdef _JACOBI
#include "jacobi.h"
#endif

#ifdef _GAUSS_SEIDEL
#include "gauss_seidel.h"
#endif

#define N_DEFAULT 100

void dealloc_memory(double*** a, int N)
{
    free_3d(a);
}

void alloc_memory(double**** a, int N)
{
    *a = malloc_3d(N, N, N);
    if (a == NULL)
    {
        perror("array allocation failed");
        exit(-1);
    }
}

void init_start_conditions(double*** u, double*** f, int N_prime, double start_T)
{
    const double RADIATION = 200;
    const double delta = 2 / (double) (N_prime - 1.0);

    const double X_LOWER_BOUND = -1;
    const double X_UPPER_BOUND = -3.0 / 8.0;
    const double Y_LOWER_BOUND = -1;
    const double Y_UPPER_BOUND = -1.0 / 2.0;
    const double Z_LOWER_BOUND = -2.0 / 3.0;
    const double Z_UPPER_BOUND = 0;


    // Initialize f.
    for (int i = 0; i < N_prime; i++)
        for (int j = 0; j < N_prime; j++)
        {
            for (int k = 0; k < N_prime; k++)
            {
                // Convert from grid space to domain space
                double x = -1.0 + delta * i;
                double y = -1.0 + delta * j;
                double z = -1.0 + delta * k;
#ifndef VALIDATE
                if (x >= X_LOWER_BOUND && x <= X_UPPER_BOUND &&
                    y >= Y_LOWER_BOUND && y <= Y_UPPER_BOUND &&
                    z >= Z_LOWER_BOUND && z <= Z_UPPER_BOUND)
                    f[i][j][k] = RADIATION;
                else
                    f[i][j][k] = 0;
#else
                // For testing
                f[i][j][k] = 3 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
#endif

            }
        }


    // Initial guess
    memset(&u[0][0][0], start_T, N_prime * N_prime * N_prime * sizeof(double));

#ifndef VALIDATE
    int edge_index = N_prime - 1;
    // Initialize boundary points
    for (int i = 0; i < N_prime; i++)
        for (int j = 0; j < N_prime; j++)
        {
            // No need to convert to domain space here,
            // since we only need predefined values at edges.
            u[i][0][j] = 0;
            u[i][edge_index][j] = 20;

            u[0][i][j] = 20;
            u[edge_index][i][j] = 20;

            u[i][j][0] = 20;
            u[i][j][edge_index] = 20;
        }
#endif


}

void validate(double*** u, int N_prime)
{
    const double delta = 1 / (double) (N_prime - 1.0);

    double total_error = 0;
    for (int i = 0; i < N_prime; i++)
        for (int j = 0; j < N_prime; j++)
        {
            for (int k = 0; k < N_prime; k++)
            {
                double x = -1.0 + delta * i;
                double y = -1.0 + delta * j;
                double z = -1.0 + delta * k;
                // For testing
                double u_ijk = sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
                //printf("%f %f\n", u[i][j][k], u_ijk);
                total_error += fabs(u[i][j][k] - u_ijk);

            }
        }

    printf("error = %f\n", total_error);
}

int main(int argc, char** argv)
{
    double start = omp_get_wtime();

    int 	N = N_DEFAULT;
    int 	N_prime = N_DEFAULT + 2;
    int 	iter_max = 1000;
    double	tolerance;
    double	start_T;
    int		output_type = 0;
    char	*output_prefix = "poisson_res";
    char        *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    double*** u = NULL;
    double*** u_old = NULL;
    double*** f = NULL;

    // 1. Input from command-line
    N         = atoi(argv[1]);	// grid size
    N_prime   = N + 2;          // For boundary points
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[4]);  // start T for all inner grid points
    if (argc == 6)
        output_type = atoi(argv[5]);  // ouput type


    // 2. Allocate memory
    alloc_memory(&u, N_prime);
    alloc_memory(&f, N_prime);

#ifdef _JACOBI
    alloc_memory(&u_old, N_prime);
#endif


    // 3. Init. f and boundary conditions
    init_start_conditions(u, f, N_prime, start_T);

    // 4. Call iterator
#ifdef _JACOBI
    jacobi(u, u_old, f, N_prime, iter_max, tolerance);
#elif defined(_GAUSS_SEIDEL)
    gauss_seidel(u, f, N_prime, iter_max, tolerance);
#endif



#ifdef VALIDATE
    validate(u, N_prime);
#endif

    // 5. Print results
    switch(output_type) {
	case 0:
	    // no output at all
	    break;
	case 3:
	    output_ext = ".bin";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write binary dump to %s: ", output_filename);
	    print_binary(output_filename, N, u);
	    break;
	case 4:
	    output_ext = ".vtk";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write VTK file to %s: ", output_filename);
	    print_vtk(output_filename, N_prime, u);
	    break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // 6. Deallocate memory
    dealloc_memory(u, N_prime);
    dealloc_memory(f, N_prime);

#ifdef _JACOBI
    dealloc_memory(u_old, N_prime);
#endif

    double elapsed = omp_get_wtime() - start;
    printf("time = %f\n", elapsed);

    return(0);
}

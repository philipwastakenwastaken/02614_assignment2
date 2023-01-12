/* jacobi.c - Poisson problem in 3d
 *
 */
#include <math.h>
#include <string.h>
#include <stdio.h>

double jacobi_update_point(double*** u_old, double*** f, double* norm_scalar, int i, int j, int k, double delta_squared)
{
    double u1 = u_old[i - 1][j][k];
    double u2 = u_old[i + 1][j][k];
    double u3 = u_old[i][j - 1][k];
    double u4 = u_old[i][j + 1][k];
    double u5 = u_old[i][j][k - 1];
    double u6 = u_old[i][j][k + 1];
    double f_term = delta_squared * f[i][j][k];

    // Iteratively calculate Frobenius-norm.
    double u_new = (u1 + u2 + u3 + u4 + u5 + u6 + f_term) / 6.0;
    double diff = u_new - u_old[i][j][k];
    *norm_scalar += diff * diff;

    return u_new;
}

void jacobi(double*** u, double*** u_old, double*** f, int N, int max_iter, double tolerance)
{
    double delta = 2 / (double) (N - 1);
    double delta_squared = delta * delta;
    double N_cubed = N * N * N;

    double d = INFINITY;
    int n = 0;

    double norm_scalar;
    double norm_scalar_part;


    while (d > tolerance && n < max_iter)
    {

        double*** temp = u;
        u = u_old;
        u_old = temp;
        norm_scalar = 0;
            

        #pragma omp parallel \
                shared(norm_scalar, u, u_old) \
                private(norm_scalar_part) \
                firstprivate(delta_squared, N_cubed, f, N)
        {

            norm_scalar_part = 0;

            #pragma omp for
            for (int i = 1; i < N - 1; i++)
                for (int j = 1; j < N - 1; j++)
                    for (int k = 1; k < N - 1; k++)
                        u[i][j][k] = jacobi_update_point(u_old, f, &norm_scalar_part, i, j, k, delta_squared);

            #pragma omp critical
            norm_scalar += norm_scalar_part;

        }

            d = sqrt(norm_scalar);
            n++;

#ifdef VERBOSE
        if (n % 100 == 0)
            printf("d = %lf iter = %d\n", d, n);
#endif

    }

#ifdef VERBOSE
    printf("Iterations: %d\n", n);
#endif
}
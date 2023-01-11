/* gauss_seidel.c - Poisson problem in 3d
 *
 */
#include <math.h>
#include <stdio.h>

double gauss_seidel_update_point(double*** u, double*** f, double* norm_scalar, int i, int j, int k, double delta_squared)
{
    double u1 = u[i - 1][j][k];
    double u2 = u[i + 1][j][k];
    double u3 = u[i][j - 1][k];
    double u4 = u[i][j + 1][k];
    double u5 = u[i][j][k - 1];
    double u6 = u[i][j][k + 1];
    double f_term = delta_squared * f[i][j][k];

    // Iteratively calculate Frobenius-norm.
    double u_new = (u1 + u2 + u3 + u4 + u5 + u6 + f_term) / 6.0;
    double diff = u_new - u[i][j][k];
    *norm_scalar += diff * diff;

    return u_new;
}


void gauss_seidel(double*** u, double*** f, int N, int max_iter, double tolerance)
{
    double delta = 1 / (double) N;
    double delta_squared = delta * delta;
    double N_cubed = N * N * N;

    double d = INFINITY;
    int n = 0;

    double norm_scalar;
    double norm_scalar_part;

    #pragma omp parallel \
            shared(norm_scalar, n, d) \
            private(norm_scalar_part) \
            firstprivate(delta_squared, N_cubed, max_iter, tolerance, N, f, u)
    {
        while (d > tolerance && n < max_iter)
        {
            norm_scalar = 0;
            norm_scalar_part = 0;

            #pragma omp for
            for (int i = 1; i < N - 1; i++)
                for (int j = 1; j < N - 1; j++)
                    for (int k = 1; k < N - 1; k++)
                        u[i][j][k] = gauss_seidel_update_point(u, f, &norm_scalar, i, j, k, delta_squared);

            #pragma omp critical
            norm_scalar += norm_scalar_part;

            #pragma omp single
            {
                d = sqrt(norm_scalar) / N_cubed;
                n++;
            }

#ifdef VERBOSE
            if (n % 100 == 0)
                printf("d = %f iter = %d\n", d, n);
#endif
        }
    }

#ifdef VERBOSE
    printf("Iterations: %d\n", n);
#endif
}


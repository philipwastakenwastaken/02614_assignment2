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
    double delta = 1 / (double) N;
    double delta_squared = delta * delta;
    double N_cubed = N * N * N;

    double d = INFINITY;
    int n = 0;

    while (d > tolerance && n < max_iter)
    {
        double norm_scalar = 0;
        // u_old = old
        memcpy(&u_old[0][0][0], &u[0][0][0], N * N * N * sizeof(double));

        for (int i = 1; i < N - 1; i++)
            for (int j = 1; j < N - 1; j++)
                for (int k = 1; k < N - 1; k++)
                    u[i][j][k] = jacobi_update_point(u_old, f, &norm_scalar, i, j, k, delta_squared);

        d = sqrt(norm_scalar) / N_cubed;

#ifdef VERBOSE
        if (n % 100 == 0)
            printf("d = %f iter = %d\n", d, n);
#endif

        n++;
    }

#ifdef VERBOSE
    printf("Iterations: %d\n", n);
#endif
}
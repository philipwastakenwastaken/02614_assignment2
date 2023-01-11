/* gauss_seidel.h - Poisson problem
 *
 */
#ifndef _GAUSS_SEIDEL_H
#define _GAUSS_SEIDEL_H

void gauss_seidel(double*** u, double*** f, int N, int max_iter, double tolerance);
double gauss_seidel_update_point(double*** u, double*** f, double* norm_scalar, int i, int j, int k, double delta_squared);

#endif

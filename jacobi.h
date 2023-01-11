/* jacobi.h - Poisson problem
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_H
#define _JACOBI_H

void jacobi(double*** u, double*** u_prime, double*** f, int N, int max_iter, double tolerance);
double fro_norm_diff(double*** a, double*** b, int N);
double jacobi_update_point(double*** u_old, double*** f, int i, int j, int k, double delta_squared);

#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "gaussian.h"

#define NITER 2395
#define SZ 60000


inline double* gaussianFilter(double* out, double* x, double sigma, double sz)
{
    for (int it = 0; it < sz; it++) {
        //double d2 = pow(x[it], 2.0);
        double d2 = x[it];
        out[it] = exp(-d2 / 2.0 / (sigma * sigma));
    }
    return out;
}


double* allocateVector(int sz)
{
    double* res = malloc(sz * sizeof(double));
    for (int it = 0; it < sz; it++)
        res[it] = it;
    return res;
}

int main(int argc, char* argv[])
{
    double* d2 = allocateVector(SZ);
    double mu_x = 1.0;
    double mu_y = 2.0;
    double sigma = 1.3;
    double* res = malloc(SZ * sizeof(double));

    for (int it = 0; it < NITER; it++) {
        gaussianFilter(res, d2, sigma, SZ);
    }

    printf("%f\n", res[0]);
    free(d2);
    free(res);
    return 0;
}

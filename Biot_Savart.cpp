#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

typedef struct { double x, y, z;} Vec3d;

void randomizeVec3d(Vec3d *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i].x = randn();
    data[i].y = randn();
    data[i].z = randn();
  }
}

void biot_savart_B(int num_points, int num_quad_points, Vec3d *points, Vec3d *gamma, Vec3d *dgamma_by_dphi, Vec3d *B) {
    for (int i = 0; i < num_points; i++) {
        // initialize
        B[i].x = 0.;
        B[i].y = 0.;
        B[i].z = 0.;
        for (int j = 0; j < num_quad_points; j++) {
            // compute the vector from target to source
            double diff_x = points[i].x - gamma[j].x;
            double diff_y = points[i].y - gamma[j].y;
            double diff_z = points[i].z - gamma[j].z;
            // compute distance between target and source
            double distSqr = diff_x*diff_x + diff_y*diff_y + diff_y*diff_y;
            double norm_diff = sqrt(diff);
            double invDist3 = 1. / (norm_diff * norm_diff * norm_diff);
            // compute cross product and reweight using distance
            B[i].x += invDist3 * (dgamma_by_dphi[j].y * diff_z - dgamma_by_dphi[j].z * diff_y);
            B[i].y += invDist3 * (dgamma_by_dphi[j].z * diff_x - dgamma_by_dphi[j].x * diff_z);
            B[i].z += invDist3 * (dgamma_by_dphi[j].x * diff_y - dgamma_by_dphi[j].y * diff_x);
        }
    }
}


int main(const int argc, const char** argv) {

  // set values
  int ntargets = 10000;
  int nsources = 20000;
  int repeat = 100;
  if (argc > 2) {
    ntargets = atoi(argv[1]);
    nsources = atoi(argv[2]);
  }

  // allocate memory
  Vec3d *points = (Vec3d*) malloc(3 * ntargets * sizeof(double));
  Vec3d *gamma = (Vec3d*) malloc(3 * nsources * sizeof(double));
  Vec3d *dgamma_by_dphi = (Vec3d*) malloc(3 * nsources * sizeof(double));
  Vec3d *B = (Vec3d*) malloc(3 * ntargets * sizeof(double));

  //initialize with randomized data
  randomizeVec3d(points, ntargets);
  randomizeVec3d(gamma, nsources);
  randomizeVec3d(dgamma_by_dphi, nsources);
  randomizeVec3d(B, ntargets);

  // CPU computation
  Timer t;
  t.tic();
  for (long i = 0; i < repeat; i++) {
    biot_savart_B(int num_points, int num_quad_points, Vec3d *points, Vec3d *gamma, Vec3d *dgamma_by_dphi, Vec3d *B);
  }
  double tt = t.toc();
  printf("CPU time = %fs\n", tt);



  // free memory
  free(points);
  free(gamma);
  free(dgamma_by_dphi);
  free(B);
}

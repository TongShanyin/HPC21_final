#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#define rand01() ((double)rand()/RAND_MAX)
#define BLOCK_SIZE 256

typedef struct { double x, y, z;} Vec3d;

void randomizeVec3d(Vec3d *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i].x = rand01();
    data[i].y = rand01();
    data[i].z = rand01();
  }
}

double errorVec3d(Vec3d *data1, Vec3d *data2, int n) {
  double err = 0.;
  for (int i = 0; i < n; i++) {
    err = std::max(err, fabs(data1[i].x -data2[i].x));
    err = std::max(err, fabs(data1[i].y -data2[i].y));
    err = std::max(err, fabs(data1[i].z -data2[i].z));
  }
  return err;
}

void CPU_biot_savart_B(int num_points, int num_quad_points, Vec3d *points, Vec3d *gamma, Vec3d *dgamma_by_dphi, Vec3d *B) {
    #pragma omp parallel for schedule(dynamic)
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
            double norm_diff = sqrt(distSqr);
            double invDist3 = 1. / (norm_diff * norm_diff * norm_diff);
            // compute cross product and reweight using distance
            B[i].x += invDist3 * (dgamma_by_dphi[j].y * diff_z - dgamma_by_dphi[j].z * diff_y);
            B[i].y += invDist3 * (dgamma_by_dphi[j].z * diff_x - dgamma_by_dphi[j].x * diff_z);
            B[i].z += invDist3 * (dgamma_by_dphi[j].x * diff_y - dgamma_by_dphi[j].y * diff_x);
        }
    }
}

__global__ void GPU_nosmem_biot_savart_B(int num_points, int num_quad_points, Vec3d *points, Vec3d *gamma, Vec3d *dgamma_by_dphi, Vec3d *B) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_points) {
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
            double norm_diff = sqrt(distSqr);
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
  int ntargets = 100;
  int nsources = 200;
  int repeat = 1;
  if (argc > 2) {
    ntargets = atoi(argv[1]);
    nsources = atoi(argv[2]);
  }

  // allocate memory
  int bytes_targets = 3 * ntargets * sizeof(double);
  int bytes_sources = 3 * nsources * sizeof(double);
  // CPU memory
  Vec3d *points = (Vec3d*) malloc(bytes_targets);
  Vec3d *gamma = (Vec3d*) malloc(bytes_sources);
  Vec3d *dgamma_by_dphi = (Vec3d*) malloc(bytes_sources);
  Vec3d *B = (Vec3d*) malloc(bytes_targets); // CPU computation
  Vec3d *B1 = (Vec3d*) malloc(bytes_targets); // GPU nosmem computation
  // GPU memory
  Vec3d *gpu_points, *gpu_gamma, *gpu_dgamma_by_dphi, *gpu_B;
  cudaMalloc(&gpu_points, bytes_targets);
  cudaMalloc(&gpu_gamma, bytes_sources);
  cudaMalloc(&gpu_dgamma_by_dphi, bytes_sources);
  cudaMalloc(&gpu_B, bytes_targets);
  int nBlocks = (ntargets + BLOCK_SIZE - 1) / BLOCK_SIZE;

  //initialize with randomized data
  randomizeVec3d(points, ntargets);
  randomizeVec3d(gamma, nsources);
  randomizeVec3d(dgamma_by_dphi, nsources);
  randomizeVec3d(B, ntargets);

  // copy data to GPU
  cudaMemcpy(gpu_points, points, bytes_targets, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_gamma, gamma, bytes_sources, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_dgamma_by_dphi, dgamma_by_dphi, bytes_sources, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_B, B, bytes_targets, cudaMemcpyHostToDevice);
  
  // CPU computation
  Timer t;
  t.tic();
  for (long i = 0; i < repeat; i++) {
    CPU_biot_savart_B(ntargets, nsources, points, gamma, dgamma_by_dphi, B);
  }
  double tt = t.toc();
  printf("CPU time = %fs\n", tt);

  // GPU computation
  t.tic();
  for (long i = 0; i < repeat; i++) {
    GPU_nosmem_biot_savart_B<<<nBlocks, BLOCK_SIZE>>>(ntargets, nsources, gpu_points, gpu_gamma, gpu_dgamma_by_dphi, gpu_B);   
  }
  tt = t.toc();
  printf("GPU no smem time = %fs\n", tt);

  // print error
  cudaMemcpy(B1, gpu_B, bytes_targets, cudaMemcpyDeviceToHost);
  double err = errorVec3d(B, B1, ntargets);
  printf("Error = %e\n", err);

  // free memory
  free(points);
  free(gamma);
  free(dgamma_by_dphi);
  free(B);
  free(B1);
  cudaFree(gpu_points);
  cudaFree(gpu_gamma);
  cudaFree(gpu_dgamma_by_dphi);
  cudaFree(gpu_B);
}

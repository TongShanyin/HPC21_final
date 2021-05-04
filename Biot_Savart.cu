#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
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

double errorVec3d_LInf(Vec3d *data1, Vec3d *data2, int n) {
  double err = 0.;
  for (int i = 0; i < n; i++) {
    err = std::max(err, fabs(data1[i].x -data2[i].x));
    err = std::max(err, fabs(data1[i].y -data2[i].y));
    err = std::max(err, fabs(data1[i].z -data2[i].z));
  }
  return err;
}

double errorVec3d_L2(Vec3d *data1, Vec3d *data2, int n) {
  double err = 0.;
  for (int i = 0; i < n; i++) {
    err += pow(data1[i].x -data2[i].x, 2);
    err += pow(data1[i].y -data2[i].y, 2);
    err += pow(data1[i].z -data2[i].z, 2);
  }
  return sqrt(err);
}

void CPU_biot_savart_B(int num_points, int num_quad_points, Vec3d *points, Vec3d *gamma, Vec3d *dgamma_by_dphi, Vec3d *B) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_points; i++) {
        // initialize
        B[i].x = 0.;
        B[i].y = 0.;
        B[i].z = 0.;
        for (int j = 0; j < num_quad_points; j++) {
            // compute the vector from target to source (6 flop)
            double diff_x = points[i].x - gamma[j].x;
            double diff_y = points[i].y - gamma[j].y;
            double diff_z = points[i].z - gamma[j].z;
            // compute distance between target and source (9 flop)
            double distSqr = diff_x*diff_x + diff_y*diff_y + diff_y*diff_y;
            double norm_diff = sqrt(distSqr);
            double invDist3 = 1. / (norm_diff * norm_diff * norm_diff);
            // compute cross product and reweight using distance (15 flop)
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

__global__ void GPU_biot_savart_B(int num_points, int num_quad_points, Vec3d *points, Vec3d *gamma, Vec3d *dgamma_by_dphi, Vec3d *B) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_points) {
        // initialize
        B[i].x  = 0.;
        B[i].y  = 0.;
        B[i].z  = 0.;
        for (int tile = 0; tile < gridDim.x; tile++) {
          // shared memory
          __shared__ Vec3d share_gamma[BLOCK_SIZE];
          __shared__ Vec3d share_dgamma_by_dphi[BLOCK_SIZE];
	  share_gamma[threadIdx.x] = gamma[tile*blockDim.x + threadIdx.x];
          share_dgamma_by_dphi[threadIdx.x] = dgamma_by_dphi[tile*blockDim.x + threadIdx.x];
	  __syncthreads();

	  // In block, compute B-S
          for (int j = 0; j < BLOCK_SIZE; j++){
            // compute the vector from target to source
            double diff_x = points[i].x - share_gamma[j].x;
            double diff_y = points[i].y - share_gamma[j].y;
            double diff_z = points[i].z - share_gamma[j].z;
            // compute distance between target and source
            double distSqr = diff_x*diff_x + diff_y*diff_y + diff_y*diff_y;
            double norm_diff = sqrt(distSqr);
            double invDist3 = 1. / (norm_diff * norm_diff * norm_diff);
            // compute cross product and reweight using distance
            B[i].x += invDist3 * (share_dgamma_by_dphi[j].y * diff_z - share_dgamma_by_dphi[j].z * diff_y);
            B[i].y += invDist3 * (share_dgamma_by_dphi[j].z * diff_x - share_dgamma_by_dphi[j].x * diff_z);
            B[i].z += invDist3 * (share_dgamma_by_dphi[j].x * diff_y - share_dgamma_by_dphi[j].y * diff_x);
    	  }
	  __syncthreads();    
        }
    }
}

int main(const int argc, const char** argv) {

  // set values
  long ntargets = 100;
  long nsources = 200;
  long repeat = 1;
  if (argc > 3) {
    ntargets = atoi(argv[1]);
    nsources = atoi(argv[2]);
    repeat = atoi(argv[3]);
  }

  // allocate memory
  long bytes_targets = 3 * ntargets * sizeof(double);
  long bytes_sources = 3 * nsources * sizeof(double);
  // CPU memory
  Vec3d *points = (Vec3d*) malloc(bytes_targets);
  Vec3d *gamma = (Vec3d*) malloc(bytes_sources);
  Vec3d *dgamma_by_dphi = (Vec3d*) malloc(bytes_sources);
  Vec3d *B = (Vec3d*) malloc(bytes_targets); // CPU computation
  Vec3d *B1 = (Vec3d*) malloc(bytes_targets); // GPU nosmem computation
  Vec3d *B2 = (Vec3d*) malloc(bytes_targets); // GPU smem computation

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
  printf("CPU flops = %3.3f GFlop/s\n", repeat * 30*ntargets*nsources/tt/1e9);
  printf("CPU Bandwidth = %3.3f GB/s\n", repeat*(3*bytes_targets+2*ntargets*bytes_sources)/ tt /1e9);
 
  // GPU nonsmem computation
  cudaDeviceSynchronize();
  t.tic();
  for (long i = 0; i < repeat; i++) {
    GPU_nosmem_biot_savart_B<<<nBlocks, BLOCK_SIZE>>>(ntargets, nsources, gpu_points, gpu_gamma, gpu_dgamma_by_dphi, gpu_B);   
  }
  cudaDeviceSynchronize();
  tt = t.toc();
  printf("GPU no smem time = %fs\n", tt);
  printf("GPU no smem flops = %3.3f GFlop/s\n", repeat * 30*ntargets*nsources/tt/1e9);
  printf("GPU no smem Bandwidth = %3.3f GB/s\n", repeat*(3*bytes_targets+2*ntargets*bytes_sources)/ tt /1e9);
  cudaMemcpy(B1, gpu_B, bytes_targets, cudaMemcpyDeviceToHost);


  // GPU computation
  cudaDeviceSynchronize();
  t.tic();
  for (long i = 0; i < repeat; i++) {
    GPU_biot_savart_B<<<nBlocks, BLOCK_SIZE>>>(ntargets, nsources, gpu_points, gpu_gamma, gpu_dgamma_by_dphi, gpu_B);
  }
  cudaDeviceSynchronize();
  tt = t.toc();
  printf("GPU time = %fs\n", tt);
  printf("GPU flops = %3.3f GFlop/s\n", repeat * 30*ntargets*nsources/tt/1e9);
  printf("GPU Bandwidth = %3.3f GB/s\n", repeat*(3*bytes_targets+2*ntargets*bytes_sources)/ tt /1e9);
  cudaMemcpy(B2, gpu_B, bytes_targets, cudaMemcpyDeviceToHost);

  // print error
  double err_LInf_1 = errorVec3d_LInf(B, B1, ntargets);
  double err_L2_1 = errorVec3d_L2(B, B1, ntargets);
  printf("nosmem LInf Error = %e, L2 Error = %e\n", err_LInf_1, err_L2_1);
  double err_LInf_2 = errorVec3d_LInf(B, B2, ntargets);
  double err_L2_2 = errorVec3d_L2(B, B2, ntargets);
  printf("LInf Error = %e, L2 Error = %e\n", err_LInf_2, err_L2_2);

  //// print some results
  //for (int i = 0; i < 10; i++){
  //  printf(" %e, %e | %e, %e | %e, %e\n", B[i].x, B1[i].x, B[i].y, B1[i].y, B[i].z, B1[i].z);
  //}

  // free memory
  free(points);
  free(gamma);
  free(dgamma_by_dphi);
  free(B);
  free(B1);
  free(B2);
  cudaFree(gpu_points);
  cudaFree(gpu_gamma);
  cudaFree(gpu_dgamma_by_dphi);
  cudaFree(gpu_B);
}

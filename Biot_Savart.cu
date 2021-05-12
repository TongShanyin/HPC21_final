#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "utils.h"

#define BLOCK_SIZE 256

typedef struct { double x, y, z;} Vec3d;

void randomizeVec3d(Vec3d *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i].x = drand48();
    data[i].y = drand48();
    data[i].z = drand48();
  }
}

double errorVec3d_LInf(Vec3d *data1, Vec3d *data2, int n) {
  double err = 0.;
  double abv = 0.;
  for (int i = 0; i < n; i++) {
    err = std::max(err, fabs(data1[i].x -data2[i].x));
    err = std::max(err, fabs(data1[i].y -data2[i].y));
    err = std::max(err, fabs(data1[i].z -data2[i].z));
    abv = std::max(abv, fabs(data1[i].x));
    abv = std::max(abv, fabs(data1[i].y));
    abv = std::max(abv, fabs(data1[i].z));
  }
  return err/abv;
}

double errorVec3d_L2(Vec3d *data1, Vec3d *data2, int n) {
  double err = 0.;
  double abv = 0.;
  for (int i = 0; i < n; i++) {
    err += pow(data1[i].x -data2[i].x, 2);
    err += pow(data1[i].y -data2[i].y, 2);
    err += pow(data1[i].z -data2[i].z, 2);
    abv += pow(data1[i].x, 2);
    abv += pow(data1[i].y, 2);
    abv += pow(data1[i].z, 2);
  }
  return sqrt(err)/sqrt(abv);
}

void CPU_biot_savart_B(int num_points, int num_quad_points, Vec3d *points, Vec3d *gamma, Vec3d *dgamma_by_dphi, Vec3d *B) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_points; i++) {
        // initialize
        double B_x = 0.;
        double B_y = 0.;
        double B_z = 0.;
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
            B_x += invDist3 * (dgamma_by_dphi[j].y * diff_z - dgamma_by_dphi[j].z * diff_y);
            B_y += invDist3 * (dgamma_by_dphi[j].z * diff_x - dgamma_by_dphi[j].x * diff_z);
            B_z += invDist3 * (dgamma_by_dphi[j].x * diff_y - dgamma_by_dphi[j].y * diff_x);
        }
        B[i].x = B_x; B[i].y = B_y; B[i].z = B_z;
    }
}

__global__ void GPU_nosmem_biot_savart_B(int num_points, int num_quad_points, Vec3d *points, Vec3d *gamma, Vec3d *dgamma_by_dphi, Vec3d *B) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_points) {
        // initialize
        double B_x = 0.;
        double B_y = 0.;
        double B_z = 0.;
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
            B_x += invDist3 * (dgamma_by_dphi[j].y * diff_z - dgamma_by_dphi[j].z * diff_y);
            B_y += invDist3 * (dgamma_by_dphi[j].z * diff_x - dgamma_by_dphi[j].x * diff_z);
            B_z += invDist3 * (dgamma_by_dphi[j].x * diff_y - dgamma_by_dphi[j].y * diff_x);
        }
	B[i].x = B_x; B[i].y = B_y; B[i].z = B_z;
    }
}

// GPU version, shared memory,ntargets == nsources are multiples of BLOCK_SIZE 
__global__ void GPU_biot_savart_B(int num_points, int num_quad_points, Vec3d *points, Vec3d *gamma, Vec3d *dgamma_by_dphi, Vec3d *B) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_points) {
        // initialize
        double B_x  = 0.;
        double B_y  = 0.;
        double B_z  = 0.;
        for (int tile = 0; tile < gridDim.x; tile++) {
          // shared memory
          __shared__ Vec3d share_gamma[BLOCK_SIZE];
          __shared__ Vec3d share_dgamma_by_dphi[BLOCK_SIZE];
	  share_gamma[threadIdx.x] = gamma[tile*blockDim.x + threadIdx.x];
          share_dgamma_by_dphi[threadIdx.x] = dgamma_by_dphi[tile*blockDim.x + threadIdx.x];
	  __syncthreads();

	  // In block, compute B-S
	  //#pragma unroll
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
            B_x += invDist3 * (share_dgamma_by_dphi[j].y * diff_z - share_dgamma_by_dphi[j].z * diff_y);
            B_y += invDist3 * (share_dgamma_by_dphi[j].z * diff_x - share_dgamma_by_dphi[j].x * diff_z);
            B_z += invDist3 * (share_dgamma_by_dphi[j].x * diff_y - share_dgamma_by_dphi[j].y * diff_x);
    	  }
	  __syncthreads();    
        }
	B[i].x = B_x; B[i].y = B_y; B[i].z = B_z;
    }
}

// GPU no shared memory, use rsqrt
__global__ void GPU_rsqrt_nosmem_biot_savart_B(int num_points, int num_quad_points, Vec3d *points, Vec3d *gamma, Vec3d *dgamma_by_dphi, Vec3d *B) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_points) {
        // initialize
        double B_x = 0.;
        double B_y = 0.;
        double B_z = 0.;
        for (int j = 0; j < num_quad_points; j++) {
            // compute the vector from target to source
            double diff_x = points[i].x - gamma[j].x;
            double diff_y = points[i].y - gamma[j].y;
            double diff_z = points[i].z - gamma[j].z;
            // compute distance between target and source
            double distSqr = diff_x*diff_x + diff_y*diff_y + diff_y*diff_y;
            double inv_norm_diff = rsqrt(distSqr);            
            double invDist3 = inv_norm_diff*inv_norm_diff*inv_norm_diff;
            // compute cross product and reweight using distance
            B_x += invDist3 * (dgamma_by_dphi[j].y * diff_z - dgamma_by_dphi[j].z * diff_y);
            B_y += invDist3 * (dgamma_by_dphi[j].z * diff_x - dgamma_by_dphi[j].x * diff_z);
            B_z += invDist3 * (dgamma_by_dphi[j].x * diff_y - dgamma_by_dphi[j].y * diff_x);
        }
        B[i].x = B_x; B[i].y = B_y; B[i].z = B_z;
    }
}

// GPU version, using rsqrt, shared memory,ntargets == nsources are multiples of BLOCK_SIZE 
__global__ void GPU_rsqrt_biot_savart_B(int num_points, int num_quad_points, Vec3d *points, Vec3d *gamma, Vec3d *dgamma_by_dphi, Vec3d *B) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_points) {
        // initialize
        double B_x  = 0.;
        double B_y  = 0.;
        double B_z  = 0.;
        for (int tile = 0; tile < gridDim.x; tile++) {
          // shared memory
          __shared__ Vec3d share_gamma[BLOCK_SIZE];
          __shared__ Vec3d share_dgamma_by_dphi[BLOCK_SIZE];
          share_gamma[threadIdx.x] = gamma[tile*blockDim.x + threadIdx.x];
          share_dgamma_by_dphi[threadIdx.x] = dgamma_by_dphi[tile*blockDim.x + threadIdx.x];
          __syncthreads();

          // In block, compute B-S
          //#pragma unroll
          for (int j = 0; j < BLOCK_SIZE; j++){
            // compute the vector from target to source
            double diff_x = points[i].x - share_gamma[j].x;
            double diff_y = points[i].y - share_gamma[j].y;
            double diff_z = points[i].z - share_gamma[j].z;
            // compute distance between target and source
            double distSqr = diff_x*diff_x + diff_y*diff_y + diff_y*diff_y;
            double inv_norm_diff = rsqrt(distSqr);
            double invDist3 = inv_norm_diff*inv_norm_diff*inv_norm_diff;
            // compute cross product and reweight using distance
            B_x += invDist3 * (share_dgamma_by_dphi[j].y * diff_z - share_dgamma_by_dphi[j].z * diff_y);
            B_y += invDist3 * (share_dgamma_by_dphi[j].z * diff_x - share_dgamma_by_dphi[j].x * diff_z);
            B_z += invDist3 * (share_dgamma_by_dphi[j].x * diff_y - share_dgamma_by_dphi[j].y * diff_x);
          }
          __syncthreads();
        }
        B[i].x = B_x; B[i].y = B_y; B[i].z = B_z;
    }
}

int main(const int argc, const char** argv) {

  // set values
  const long PFIRST = pow(2, 10);
  const long PLAST = pow(2, 16);
  FILE * pFile;
  pFile = fopen ("data.txt","w");

  Timer t;
  fprintf(pFile, "       p       ");
  fprintf(pFile, "t_CPU     t_GPU_ns    t_GPU    t_ns_rs    t_rs    ");
  fprintf(pFile, "flops_CPU  flops_GPU_ns flops_GPU flops_ns_rs  flops_rs  ");
  fprintf(pFile, "band_CPU  band_GPU_ns  band_GPU band_ns_rs   band_rs  ");
  //fprintf(pFile, "err_GPU_ns  err_GPU err_ns_rs  err_rs");
  fprintf(pFile, "\n");
  // allocate memory
for (long p = PFIRST; p <= PLAST; p *= 2) {
  long nsources = p, ntargets = p;
  long repeat = 1e8/(nsources*ntargets)+1;
  long bytes_targets = 3 * ntargets * sizeof(double);
  long bytes_sources = 3 * nsources * sizeof(double);
  // CPU memory
  Vec3d *points = (Vec3d*) malloc(bytes_targets);
  Vec3d *gamma = (Vec3d*) malloc(bytes_sources);
  Vec3d *dgamma_by_dphi = (Vec3d*) malloc(bytes_sources);
  Vec3d *B = (Vec3d*) malloc(bytes_targets); // CPU computation
  Vec3d *B1 = (Vec3d*) malloc(bytes_targets); // GPU computation


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
  //cudaMemcpy(gpu_points, points, bytes_targets, cudaMemcpyHostToDevice);
  //cudaMemcpy(gpu_gamma, gamma, bytes_sources, cudaMemcpyHostToDevice);
  //cudaMemcpy(gpu_dgamma_by_dphi, dgamma_by_dphi, bytes_sources, cudaMemcpyHostToDevice);
  //cudaMemcpy(gpu_B, B, bytes_targets, cudaMemcpyHostToDevice);
  
  // CPU computation
  t.tic();
  for (long i = 0; i < repeat; i++) {
    CPU_biot_savart_B(ntargets, nsources, points, gamma, dgamma_by_dphi, B);
  }
  double tt = t.toc();
  double fp = repeat * 30*ntargets*nsources/tt/1e9;
  double bd = repeat*(2*bytes_targets+2*ntargets*bytes_sources)/ tt /1e9;
  
  // GPU nonsmem computation
  cudaDeviceSynchronize();
  t.tic();
  for (long i = 0; i < repeat; i++) {
    cudaMemcpy(gpu_points, points, bytes_targets, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_gamma, gamma, bytes_sources, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dgamma_by_dphi, dgamma_by_dphi, bytes_sources, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, B, bytes_targets, cudaMemcpyHostToDevice);
    
    GPU_nosmem_biot_savart_B<<<nBlocks, BLOCK_SIZE>>>(ntargets, nsources, gpu_points, gpu_gamma, gpu_dgamma_by_dphi, gpu_B);   
    cudaMemcpy(B1, gpu_B, bytes_targets, cudaMemcpyDeviceToHost);
   }
  cudaDeviceSynchronize();
  double tt1 = t.toc();
  double fp1 = repeat * 30*ntargets*nsources/tt1/1e9;
  double bd1 = repeat*(2*bytes_targets+2*ntargets*bytes_sources)/ tt1 /1e9;
  double err1 = errorVec3d_LInf(B, B1, ntargets);

  // GPU computation with shared memory
  cudaDeviceSynchronize();
  t.tic();
  for (long i = 0; i < repeat; i++) {
    cudaMemcpy(gpu_points, points, bytes_targets, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_gamma, gamma, bytes_sources, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dgamma_by_dphi, dgamma_by_dphi, bytes_sources, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, B, bytes_targets, cudaMemcpyHostToDevice);

    GPU_biot_savart_B<<<nBlocks, BLOCK_SIZE>>>(ntargets, nsources, gpu_points, gpu_gamma, gpu_dgamma_by_dphi, gpu_B);

    cudaMemcpy(B1, gpu_B, bytes_targets, cudaMemcpyDeviceToHost);
  }
  cudaDeviceSynchronize();
  double tt2 = t.toc();
  double fp2 = repeat * 30*ntargets*nsources/tt2/1e9;
  double bd2 = repeat*(2*bytes_targets+2*ntargets*bytes_sources)/ tt2 /1e9;
  double err2 = errorVec3d_LInf(B, B1, ntargets);

  // GPU nonsmem computation using rsqrt
  cudaDeviceSynchronize();
  t.tic();
  for (long i = 0; i < repeat; i++) {
    cudaMemcpy(gpu_points, points, bytes_targets, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_gamma, gamma, bytes_sources, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dgamma_by_dphi, dgamma_by_dphi, bytes_sources, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, B, bytes_targets, cudaMemcpyHostToDevice);
    
    GPU_rsqrt_nosmem_biot_savart_B<<<nBlocks, BLOCK_SIZE>>>(ntargets, nsources, gpu_points, gpu_gamma, gpu_dgamma_by_dphi, gpu_B);
    cudaMemcpy(B1, gpu_B, bytes_targets, cudaMemcpyDeviceToHost);
   }
  cudaDeviceSynchronize();
  double tt3 = t.toc();
  double fp3 = repeat * 29*ntargets*nsources/tt3/1e9;
  double bd3 = repeat*(2*bytes_targets+2*ntargets*bytes_sources)/ tt3 /1e9;
  double err3 = errorVec3d_LInf(B, B1, ntargets);

  // GPU computation with shared memory, using rsqrt
  cudaDeviceSynchronize();
  t.tic();
  for (long i = 0; i < repeat; i++) {
    cudaMemcpy(gpu_points, points, bytes_targets, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_gamma, gamma, bytes_sources, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dgamma_by_dphi, dgamma_by_dphi, bytes_sources, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, B, bytes_targets, cudaMemcpyHostToDevice);

    GPU_rsqrt_biot_savart_B<<<nBlocks, BLOCK_SIZE>>>(ntargets, nsources, gpu_points, gpu_gamma, gpu_dgamma_by_dphi, gpu_B);

    cudaMemcpy(B1, gpu_B, bytes_targets, cudaMemcpyDeviceToHost);
  }
  cudaDeviceSynchronize();
  double tt4 = t.toc();
  double fp4 = repeat * 29*ntargets*nsources/tt4/1e9;
  double bd4 = repeat*(2*bytes_targets+2*ntargets*bytes_sources)/ tt4 /1e9;
  double err4 = errorVec3d_LInf(B, B1, ntargets);

  fprintf(pFile, "%10d ", p);
  fprintf(pFile, "%10f %10f %10f %10f %10f ", tt/repeat, tt1/repeat, tt2/repeat, tt3/repeat, tt4/repeat);
  fprintf(pFile, "%10f %10f %10f %10f %10f ", fp, fp1, fp2, fp3, fp4);
  fprintf(pFile, "%10f %10f %10f %10f %10f ", bd, bd1, bd2, bd3, bd4);
 // fprintf(pFile, "%10f %10f %10f %10f", err1, err2, err3, err4);
  fprintf(pFile, "\n");
  // print some results
//  for (int i = 0; i < 2; i++){
 //   printf(" %e, %e | %e, %e | %e, %e\n", B[i].x, B1[i].x, B[i].y, B1[i].y, B[i].z, B1[i].z);
  //}

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
fclose (pFile);
}

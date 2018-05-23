#include <stdio.h>

extern "C" {

__device__ __constant__ int n;
__device__ __constant__ int* offsets;
__device__ __constant__ int* outlist;
__device__ int* parents;
__device__ int* frontier;

__device__ int** next_frontier;
__device__ int*  next_len;


enum Errors {
  NONE = 0,
  MALLOC = -1,
};
__device__ int err = NONE;

__global__ void bfs() {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  int v = frontier[idx];

  int out = offsets[v];
  int len = offsets[v+1] - out;

  int  num_discovered = 0;
  int* new_frontier = (int*) malloc(len * sizeof(int)); // TODO handle malloc failure
  if (!new_frontier) {
    err = true;
    return;
  }
  for (int i = 0; i < len; i++) {
    int edge = outlist[out + i];

    int old = atomicCAS(&parents[edge], ~0, v);
    if (old == ~0) {
      new_frontier[num_discovered++] = edge;
    }
  }

  next_frontier[idx] = new_frontier;
  next_len[idx] = num_discovered;
}

__global__ void reduce(int t, int w) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx + w >= t) return;

  int* a  = next_frontier[idx];
  int  al = next_len[idx];

  int* b  = next_frontier[idx+w];
  int  bl = next_len[idx+w];

  int  cl = al + bl;
  int* c  = (int*) malloc(cl * sizeof(int)); // TODO handle malloc failure
  memcpy(c   , a, al*sizeof(int));
  memcpy(c+al, b, bl*sizeof(int));
  free(a);
  free(b);

  next_frontier[idx]   = c;
  next_frontier[idx+w] = 0;
  next_len[idx]   = al+bl;
  next_len[idx+w] = 0;
}

__global__ void step() {
  int* f = next_frontier[0];
  memcpy(frontier, f, next_len[0]*sizeof(int));
  free(f);
}

} // extern "C"

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
  if (v > n) printf("idx %d v %d n %d\n", idx, v, n);

  int out = offsets[v];
  int len = offsets[v+1] - out;

  int  num_discovered = 0;
  int* new_frontier = (int*) malloc(len * sizeof(int));
  if (!new_frontier) {
    err = true;
    return;
  }
  /* printf("v = %d, offset = %d, len = %d, parent = %d\n", v, out, len, parents[v]); */
  for (int i = 0; i < len; i++) {
    int edge = outlist[out + i];
    /* printf("edge[%d] = %d\n", i, edge); */

    int old = atomicCAS(&parents[edge], ~0, v);
    /* printf("cas %d\n", old); */
    if (old == ~0) {
      new_frontier[num_discovered++] = edge;
    }
  }

  next_frontier[idx] = new_frontier;
  next_len[idx] = num_discovered;
}

__global__ void reduce(int t, int w) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x);
  /* printf("reduce(%d, %d) idx=%d\n", t, w, idx); */
  if (idx + w >= t) return;

  /* printf("reduce %d + %d = %d\n", idx, w, idx+w); */
  int* a  = next_frontier[idx];
  int  al = next_len[idx];

  int* b  = next_frontier[idx+w];
  int  bl = next_len[idx+w];

  int  cl = al + bl;
  int* c  = (int*) malloc(cl * sizeof(int));
  if (!c) { err = MALLOC; return; }
  memcpy(c   , a, al*sizeof(int));
  memcpy(c+al, b, bl*sizeof(int));
  free(a);
  free(b);

  next_frontier[idx]   = c;
  next_frontier[idx+w] = 0;
  printf("al, bl %d %d\n", al, bl);
  next_len[idx]   = al+bl;
  next_len[idx+w] = 0;
}

__global__ void step() {
  int* f = next_frontier[0];
  memcpy(frontier, f, next_len[0]*sizeof(int));
  free(f);
}

} // extern "C"

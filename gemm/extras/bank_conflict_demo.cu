#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#define CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){printf("%s\n",cudaGetErrorString(e));exit(1);}}while(0)
constexpr int N=32;

__global__ void conflict(float* out) {
  __shared__ float s[N][N];
  int x=threadIdx.x, y=threadIdx.y;
  s[y][x] = float(y*N+x);
  __syncthreads();
  // For a warp with x varying, this reads a column: stride=32 words -> same bank, different addresses.
  float v = s[x][y];
  if (y==0) out[x]=v;
}
__global__ void padded(float* out) {
  __shared__ float s[N][N+1];
  int x=threadIdx.x, y=threadIdx.y;
  s[y][x] = float(y*N+x);
  __syncthreads();
  float v = s[x][y];
  if (y==0) out[x]=v;
}
__global__ void swizzled(float* out) {
  __shared__ float s[N][N];
  int x=threadIdx.x, y=threadIdx.y;
  int sx = x ^ y;
  s[y][sx] = float(y*N+x);
  __syncthreads();
  float v = s[x][y ^ x];
  if (y==0) out[x]=v;
}
int main(int argc,char**argv){
  float* d; CHECK(cudaMalloc(&d,N*sizeof(float)));
  dim3 b(32,32); dim3 g(1);
  const char* mode=argc>1?argv[1]:"conflict";
  for(int i=0;i<200;i++){
    if(mode==std::string("padded")) padded<<<g,b>>>(d);
    else if(mode==std::string("swizzle")) swizzled<<<g,b>>>(d);
    else conflict<<<g,b>>>(d);
  }
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaFree(d));
  return 0;
}

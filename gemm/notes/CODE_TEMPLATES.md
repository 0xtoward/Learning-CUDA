# Code templates to memorize

## 1. Coalesced lane mapping
```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
// across a warp x changes fastest -> adjacent columns
```

## 2. K-axis shared-memory tiling
```cpp
for (int k0 = 0; k0 < K; k0 += BK) {
  // cooperative GMEM -> SMEM
  As[...] = A[...];
  Bs[...] = B[...];
  __syncthreads();
  for (int k = 0; k < BK; ++k) acc += As[...][k] * Bs[k][...];
  __syncthreads();
}
```

## 3. 1D thread tiling
```cpp
float acc[TM] = {};
for (int k = 0; k < BK; ++k) {
  float b = Bs[k][col];
  for (int i = 0; i < TM; ++i) acc[i] += As[row0+i][k] * b;
}
```

## 4. 2D thread tiling / register outer product
```cpp
float acc[TM][TN] = {};
for (int k = 0; k < BK; ++k) {
  float a[TM], b[TN];
  // SMEM -> registers
  for (int i=0;i<TM;++i) a[i] = As[row0+i][k];
  for (int j=0;j<TN;++j) b[j] = Bs[k][col0+j];
  // outer product
  for (int i=0;i<TM;++i)
    for (int j=0;j<TN;++j)
      acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
}
```

## 5. Bank-conflict fixes
```cpp
// padding
__shared__ float tile[32][33];

// simple XOR swizzle
int physical_col = logical_col ^ row;
```

## 6. Vectorized global load
```cpp
float4 x = *reinterpret_cast<const float4*>(ptr); // requires alignment/in-bounds
```

## 7. Warp tiling coordinates
```cpp
int warp = threadIdx.x / 32;
int lane = threadIdx.x % 32;
int warp_m = warp / WARPS_N;
int warp_n = warp % WARPS_N;
// then map lane -> thread micro-tile inside the warp tile
```

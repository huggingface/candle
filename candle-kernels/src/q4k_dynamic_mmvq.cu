#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float q4_half(uint16_t v) { return __half2float(*reinterpret_cast<const __half*>(&v)); }
__device__ __forceinline__ float warp_sum(float v) { for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o); return v; }
__device__ __forceinline__ uint8_t sb(uint32_t a,uint32_t b,uint32_t c,int i){uint32_t w=i<4?a:(i<8?b:c);return (w>>((i&3)*8))&255;}
__device__ __forceinline__ void sm(uint32_t a,uint32_t b,uint32_t c,int n,float& s,float& m){if(n<4){s=sb(a,b,c,n)&63;m=sb(a,b,c,n+4)&63;}else{uint8_t lo=sb(a,b,c,n+4);s=(lo&15)|((sb(a,b,c,n-4)>>6&3)<<4);m=(lo>>4)|((sb(a,b,c,n)>>6&3)<<4);}}
extern "C" __global__ void q4k_dynamic_mmvq4_decode_v1(const uint8_t* w,const float* x,float* out,size_t rows,size_t blocks){
 extern __shared__ float sx[];
 int tid=threadIdx.x, warp=tid>>5, lane=tid&31; size_t base=(size_t)blockIdx.x*4, row=base+warp;
 if(warp>=4) return; float acc=0;
 for(size_t bi=0;bi<blocks;bi++){
  for(int i=tid;i<256;i+=128) sx[i]=x[bi*256+i]; __syncthreads();
  if(row<rows){const uint8_t* p=w+(row*blocks+bi)*144; uint32_t h=lane==0?*reinterpret_cast<const uint32_t*>(p):0;uint32_t a=lane==0?*reinterpret_cast<const uint32_t*>(p+4):0,b=lane==0?*reinterpret_cast<const uint32_t*>(p+8):0,c=lane==0?*reinterpret_cast<const uint32_t*>(p+12):0;
   h=__shfl_sync(0xffffffff,h,0);a=__shfl_sync(0xffffffff,a,0);b=__shfl_sync(0xffffffff,b,0);c=__shfl_sync(0xffffffff,c,0);float d=q4_half(h&65535),dm=q4_half(h>>16);const uint8_t* q=p+16;
   for(int g=0;g<4;g++){uint8_t z=q[32*g+lane];float s,m;sm(a,b,c,2*g,s,m);acc+=(d*s*(z&15)-dm*m)*sx[64*g+lane];sm(a,b,c,2*g+1,s,m);acc+=(d*s*(z>>4)-dm*m)*sx[64*g+32+lane];}}
  __syncthreads(); }
 if(row<rows){acc=warp_sum(acc);if(lane==0)out[row]=acc;}
}

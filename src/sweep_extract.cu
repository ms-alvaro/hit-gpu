#include "turH.h"

/*
 * GPU plane-extraction for the swept inflow.
 *
 * Replaces the CPU sum_over_kx (full-field D2H copy + strided host sum, which
 * left the A100 ~1% utilised).  The Fourier field is already on the GPU, so we
 * sum over kx with the per-mode phase exp(i*2*pi*m*s) directly on the device
 * and leave the (ky,kz) slab on the GPU for the following 2D C2R.
 *
 * Layout (matches the rest of the code): comp[kx*NY*NZ + ky*NZ + kz].
 * One thread per (ky,kz); threads in a warp hit consecutive kz -> coalesced.
 * Accumulate in double to match the old CPU result (~1e-7 vs full backward FFT).
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float2* phase_dev  = NULL;   /* [NX] exp(i*2*pi*m*s) per kx mode */
static float2* phase_host = NULL;

static __global__ void sum_kx_phase_kernel(const float2* __restrict comp,
                                           const float2* __restrict phase,
                                           float2* __restrict slab, int NXS)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   /* idx = ky*NZ + kz */
    if (idx >= NY * NZ) return;
    double accr = 0.0, acci = 0.0;
    for (int kx = 0; kx < NXS; kx++) {
        float2 v = comp[kx * NY * NZ + idx];
        float2 p = phase[kx];
        accr += (double)v.x * p.x - (double)v.y * p.y;
        acci += (double)v.x * p.y + (double)v.y * p.x;
    }
    slab[idx].x = (float)accr;
    slab[idx].y = (float)acci;
}

/* Phase-weighted sum over kx at streamwise fraction s_frac, on the GPU.
   comp_gpu : Fourier field component (NXSIZE*NY*NZ float2, on device)
   slab_gpu : output (NY*NZ float2, on device) */
extern void sum_over_kx_gpu(float2* comp_gpu, float2* slab_gpu,
                            double s_frac, int NXS)
{
    if (phase_dev == NULL) {
        cudaMalloc((void**)&phase_dev, NX * sizeof(float2));
        phase_host = (float2*)malloc(NX * sizeof(float2));
    }
    /* per-kx phase: m = (kx<NX/2)?kx:kx-NX ; exp(i*2*pi*m*s_frac) */
    for (int kx = 0; kx < NXS; kx++) {
        int m = (kx < NX / 2) ? kx : kx - NX;
        double ang = 2.0 * M_PI * (double)m * s_frac;
        phase_host[kx].x = (float)cos(ang);
        phase_host[kx].y = (float)sin(ang);
    }
    cudaMemcpy(phase_dev, phase_host, NXS * sizeof(float2), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (NY * NZ + threads - 1) / threads;
    sum_kx_phase_kernel<<<blocks, threads>>>(comp_gpu, phase_dev, slab_gpu, NXS);
    cudaDeviceSynchronize();
}

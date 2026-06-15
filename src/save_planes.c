/*
 * save_planes.c — Extract and save physical-space y-z planes from HIT.
 *
 * For SIZE=1 only (single GPU).
 * Uses a dedicated cuFFT 3D C2R plan to avoid touching the code's
 * overlapped FFT infrastructure.
 *
 * Fourier-space memory layout (set by the code's forward FFT):
 *   After forward:  data is in (ky, kz, kx) order with kx fastest.
 *   This matches cuFFT 3D C2R with dims (NY, 2*(NZ-1), NX) when
 *   the R2C half-dimension is the LAST one (kx).
 *
 * BUT: the code's R2C is only in z, not in x. All three indices have
 * full complex modes. So we cannot use a standard 3D C2R plan.
 *
 * Instead, we do the inverse transform in two steps:
 *   1) Batched 1D C2C inverse in kx (the fastest index)
 *   2) After reordering to (x, ky, kz) layout via CPU transpose,
 *      batched 2D C2R inverse in (ky, kz) -> (y, z) for each x
 *
 * For simplicity and correctness, we do EVERYTHING on CPU:
 *   - Copy Fourier data to CPU
 *   - Transpose (ky,kz,kx) -> (kx,ky,kz) on CPU
 *   - Use numpy-style loops for 1D IFFT in x, 2D IRFFT in (y,z)
 *
 * Actually, even simpler: for a SINGLE x-plane (ix=0), we just need
 * to SUM over all kx modes (the Fourier inverse at x=0 is a sum):
 *   u(x=0, ky, kz) = sum_{kx} uhat(ky, kz, kx)
 * Then do 2D C2R IFFT of the resulting (ky, kz) slab.
 *
 * This is by far the cheapest approach: no 3D transform needed.
 * We use a single cuFFT 2D C2R plan for the final step.
 */

#include "turH.h"

/* --- Local state --- */
static float2* sum_slab_gpu = NULL;   /* (NY, NZ) complex slab on GPU  */
static float*  plane_real_gpu = NULL; /* (NY, 2*NZ) real slab on GPU   */
static float*  plane_cpu[3];          /* (N, N) float planes on CPU    */
static cufftHandle plan_2d_c2r;      /* 2D C2R plan for one slab      */
static FILE*   plane_fp = NULL;
static int     plane_init = 0;
static int     plane_count = 0;
static double  sweep_period = 0.0;   /* HIT-time sweep period; 0 => fixed x=0 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void save_planes_init(const char* filename, float sweep_period_in)
{
    if (RANK != 0) return;

    sweep_period = (double)sweep_period_in;

    /* GPU buffers */
    CHECK_CUDART(cudaMalloc((void**)&sum_slab_gpu, NY * NZ * sizeof(float2)));
    /* The C2R output overwrites the complex buffer in-place.
       Ensure buffer is large enough: NY * 2*NZ floats = NY * NZ * 2 floats
       = NY * NZ * sizeof(float2).  So same allocation works. */

    /* CPU buffers */
    for (int c = 0; c < 3; c++)
        plane_cpu[c] = (float*)malloc(N * N * sizeof(float));

    /* cuFFT 2D C2R plan: dims = (NY, N), batch = 1 */
    int n2d[2] = {NY, N};                     /* NY = N for this code */
    cufftPlanMany(&plan_2d_c2r, 2, n2d,
                  NULL, 1, 0,                 /* input  (complex): auto strides */
                  NULL, 1, 0,                 /* output (real):    auto strides */
                  CUFFT_C2R, 1);              /* batch = 1 */

    /* Open output file and write header */
    plane_fp = fopen(filename, "wb");
    if (!plane_fp) {
        fprintf(stderr, "save_planes: cannot open %s\n", filename);
        return;
    }
    int hdr_n = N;
    float hdr_re = REYNOLDS;
    float hdr_nu = 1.0f / REYNOLDS;
    fwrite(&hdr_n,  sizeof(int),   1, plane_fp);
    fwrite(&hdr_re, sizeof(float), 1, plane_fp);
    fwrite(&hdr_nu, sizeof(float), 1, plane_fp);
    fflush(plane_fp);

    plane_init = 1;
    plane_count = 0;
    if (sweep_period > 0.0)
        printf("save_planes: initialized, N=%d, file=%s, SWEEP mode "
               "(sweep_period=%g HIT time; x_s/Lx = mod(t/sweep_period,1))\n",
               N, filename, sweep_period);
    else
        printf("save_planes: initialized, N=%d, file=%s, fixed x=0 plane\n",
               N, filename);
}

/*
 * Sum over kx modes on GPU.
 *
 * Input:  component  — float2 array of size (NXSIZE * NY * NZ), laid out
 *         with kx as the fastest-varying index: idx = ky*NZ*NXSIZE + kz*NXSIZE + kx.
 *         (This is the layout AFTER the code's forward FFT.)
 *
 * Output: sum_slab_gpu — float2 array of size (NY * NZ), where
 *         sum_slab_gpu[ky*NZ + kz] = sum_{kx=0}^{NXSIZE-1} component[ky*NZ*NXSIZE + kz*NXSIZE + kx]
 *
 * For SIZE=1, NXSIZE = NX = N.
 */
static void sum_over_kx_cpu(float2* component, double s_frac)
{
    /* Copy full Fourier field to CPU, sum, copy slab back to GPU.

       To extract the plane at streamwise fraction s_frac = x_s/Lx instead
       of x=0, evaluate the inverse x-transform at x_s by weighting each
       kx mode with the phase exp(i * kx * x_s) = exp(i * 2*pi * m * s_frac),
       where m is the integer x-wavenumber for index kx (FFT ordering):
           m = (kx < NX/2) ? kx : kx - NX.
       For SIZE=1 (IGLOBAL=0) NXSIZE = NX. s_frac=0 reproduces the x=0 sum. */
    int total = NXSIZE * NY * NZ;
    float2* host_buf = (float2*)malloc(total * sizeof(float2));
    float2* host_slab = (float2*)malloc(NY * NZ * sizeof(float2));

    CHECK_CUDART(cudaMemcpy(host_buf, component, total * sizeof(float2),
                            cudaMemcpyDeviceToHost));

    /* Precompute per-kx phase factor exp(i * 2*pi * m * s_frac) */
    double* phc = (double*)malloc(NXSIZE * sizeof(double));
    double* phs = (double*)malloc(NXSIZE * sizeof(double));
    for (int kx = 0; kx < NXSIZE; kx++) {
        int m = (kx < NX / 2) ? kx : kx - NX;
        double ang = 2.0 * M_PI * (double)m * s_frac;
        phc[kx] = cos(ang);
        phs[kx] = sin(ang);
    }

    /* Zero the slab */
    for (int i = 0; i < NY * NZ; i++) {
        host_slab[i].x = 0.0f;
        host_slab[i].y = 0.0f;
    }

    /* Phase-weighted sum over kx (slowest index).
       Memory layout: (kx, ky, kz) with kz fastest.
       full_idx = kx*NY*NZ + ky*NZ + kz
       (re + i im) * (c + i s) = (re*c - im*s) + i(re*s + im*c) */
    for (int ky = 0; ky < NY; ky++) {
        for (int kz = 0; kz < NZ; kz++) {
            int slab_idx = ky * NZ + kz;
            double acc_r = 0.0, acc_i = 0.0;
            for (int kx = 0; kx < NXSIZE; kx++) {
                int full_idx = kx * NY * NZ + ky * NZ + kz;
                double re = host_buf[full_idx].x;
                double im = host_buf[full_idx].y;
                acc_r += re * phc[kx] - im * phs[kx];
                acc_i += re * phs[kx] + im * phc[kx];
            }
            host_slab[slab_idx].x = (float)acc_r;
            host_slab[slab_idx].y = (float)acc_i;
        }
    }

    free(phc);
    free(phs);

    CHECK_CUDART(cudaMemcpy(sum_slab_gpu, host_slab, NY * NZ * sizeof(float2),
                            cudaMemcpyHostToDevice));

    free(host_buf);
    free(host_slab);
}

/*
 * Debug validation: use the code's own backward FFT to get ground truth,
 * then compare with our sum-over-kx extraction.
 * Only runs once (first call).
 */
static void validate_extraction(float2* component, float* extracted_plane)
{
    static int validated = 0;
    if (validated) return;
    validated = 1;

    float N3 = (float)N * (float)N * (float)N;
    int z_stride = 2 * NZ;
    size_t fft_size = (size_t)NXSIZE * NY * NZ * sizeof(float2);

    printf("\n=== VALIDATION: comparing extraction with full backward FFT ===\n");

    /* Allocate temp GPU buffer and copy Fourier data */
    float2* temp_gpu;
    CHECK_CUDART(cudaMalloc((void**)&temp_gpu, fft_size));
    CHECK_CUDART(cudaMemcpy(temp_gpu, component, fft_size, cudaMemcpyDeviceToDevice));

    /* Full backward FFT (modifies temp_gpu in place) */
    fftBackward(temp_gpu);
    CHECK_CUDART(cudaDeviceSynchronize());

    /* Copy x=0 plane to CPU (first NY * z_stride floats) */
    float* phys_cpu = (float*)malloc(NY * z_stride * sizeof(float));
    CHECK_CUDART(cudaMemcpy(phys_cpu, (float*)temp_gpu,
                            NY * z_stride * sizeof(float),
                            cudaMemcpyDeviceToHost));

    /* Compare: ground truth = phys_cpu / N^3, extraction = extracted_plane */
    double max_err = 0, sum_sq_err = 0, sum_sq_gt = 0;
    double mean_gt = 0, mean_ext = 0;
    double rms_gt = 0, rms_ext = 0;
    for (int iy = 0; iy < N; iy++) {
        for (int iz = 0; iz < N; iz++) {
            float gt = phys_cpu[iy * z_stride + iz] / N3;
            float ext = extracted_plane[iy * N + iz];
            double err = fabs((double)gt - (double)ext);
            if (err > max_err) max_err = err;
            sum_sq_err += err * err;
            sum_sq_gt += (double)gt * (double)gt;
            mean_gt += gt;
            mean_ext += ext;
            rms_gt += (double)gt * (double)gt;
            rms_ext += (double)ext * (double)ext;
        }
    }
    int nn = N * N;
    mean_gt /= nn;
    mean_ext /= nn;
    rms_gt = sqrt(rms_gt / nn - mean_gt * mean_gt);
    rms_ext = sqrt(rms_ext / nn - mean_ext * mean_ext);

    printf("  Ground truth (backward FFT):   mean=%+.6e  rms=%.6e\n", mean_gt, rms_gt);
    printf("  Extraction (sum-kx + C2R):     mean=%+.6e  rms=%.6e\n", mean_ext, rms_ext);
    printf("  Max absolute error:            %.6e\n", max_err);
    printf("  RMS error:                     %.6e\n", sqrt(sum_sq_err / nn));
    printf("  Relative error (err/rms_gt):   %.6e\n",
           rms_gt > 0 ? sqrt(sum_sq_err / nn) / rms_gt : 0.0);
    printf("  rms_extraction / rms_gt:       %.6f\n",
           rms_gt > 0 ? rms_ext / rms_gt : 0.0);
    printf("=== END VALIDATION ===\n\n");

    cudaFree(temp_gpu);
    free(phys_cpu);
}

/*
 * One-shot validation of the SWEPT (phase-shifted) extraction.
 * For several integer grid indices ix, the phase fraction s = ix/N must
 * reproduce exactly the physical y-z plane at grid index ix obtained from
 * the code's own full backward FFT.  Catches kx wavenumber sign/ordering
 * errors in the phase ramp.  Runs once.
 */
static void validate_sweep(float2* component)
{
    static int done = 0;
    if (done) return;
    done = 1;

    float N3 = (float)N * (float)N * (float)N;
    int z_stride = 2 * NZ;
    size_t fft_size = (size_t)NXSIZE * NY * NZ * sizeof(float2);

    float2* temp_gpu;
    CHECK_CUDART(cudaMalloc((void**)&temp_gpu, fft_size));
    CHECK_CUDART(cudaMemcpy(temp_gpu, component, fft_size, cudaMemcpyDeviceToDevice));
    fftBackward(temp_gpu);
    CHECK_CUDART(cudaDeviceSynchronize());

    /* full physical field (x slowest, then y, then padded z) */
    size_t phys_floats = (size_t)NX * NY * z_stride;
    float* phys = (float*)malloc(phys_floats * sizeof(float));
    CHECK_CUDART(cudaMemcpy(phys, (float*)temp_gpu, phys_floats * sizeof(float),
                            cudaMemcpyDeviceToHost));

    float* slab = (float*)malloc(NY * z_stride * sizeof(float));
    int tests[5] = {0, 1, 7, N/2, N-1};

    printf("\n=== SWEEP VALIDATION (phase extraction vs backward FFT) ===\n");
    for (int t = 0; t < 5; t++) {
        int ix = tests[t];
        double s = (double)ix / (double)N;
        sum_over_kx_cpu(component, s);
        cufftExecC2R(plan_2d_c2r, (cufftComplex*)sum_slab_gpu,
                     (cufftReal*)sum_slab_gpu);
        CHECK_CUDART(cudaDeviceSynchronize());
        CHECK_CUDART(cudaMemcpy(slab, (float*)sum_slab_gpu,
                                NY * z_stride * sizeof(float),
                                cudaMemcpyDeviceToHost));
        double serr = 0, sgt = 0, mx = 0;
        for (int iy = 0; iy < N; iy++)
            for (int iz = 0; iz < N; iz++) {
                double gt  = phys[(size_t)ix*NY*z_stride + iy*z_stride + iz] / N3;
                double ext = slab[iy * z_stride + iz] / N3;
                double e = fabs(gt - ext);
                if (e > mx) mx = e;
                serr += e * e; sgt += gt * gt;
            }
        printf("  ix=%4d  s=%.4f  rel_rms_err=%.3e  max_abs_err=%.3e\n",
               ix, s, sgt > 0 ? sqrt(serr / sgt) : 0.0, mx);
    }
    printf("=== END SWEEP VALIDATION ===\n\n");

    cudaFree(temp_gpu);
    free(phys);
    free(slab);
}

void save_plane_step(vectorField u, float time)
{
    if (!plane_init || RANK != 0) return;

    float N3 = (float)N * (float)N * (float)N;
    int z_stride = 2 * NZ;   /* N + 2 (C2R padding in z) */

    /* Temp CPU buffer for one real slab with z-padding */
    float* real_slab = (float*)malloc(NY * z_stride * sizeof(float));

    /* Streamwise sweep fraction s = x_s/Lx for this time instant.
       sweep_period<=0 => s=0 (legacy fixed x=0 plane). */
    double s_frac = 0.0;
    if (sweep_period > 0.0) {
        s_frac = fmod((double)time / sweep_period, 1.0);
        if (s_frac < 0.0) s_frac += 1.0;
    }

    float2* comps[3] = {u.x, u.y, u.z};

    /* One-shot self-test of the phase extraction (no-op after first call) */
    validate_sweep(u.x);

    for (int c = 0; c < 3; c++) {
        /* 1) Sum over kx (phase-weighted at x_s) to get (ky, kz) slab */
        sum_over_kx_cpu(comps[c], s_frac);

        /* 2) 2D C2R inverse: (ky, kz) -> (y, z).
              Input:  sum_slab_gpu, NY * NZ complex values
              Output: same buffer reinterpreted as NY * 2*NZ reals (in-place) */
        cufftExecC2R(plan_2d_c2r, (cufftComplex*)sum_slab_gpu,
                     (cufftReal*)sum_slab_gpu);
        CHECK_CUDART(cudaDeviceSynchronize());

        /* 3) Copy real slab (NY * z_stride floats) to CPU */
        CHECK_CUDART(cudaMemcpy(real_slab, (float*)sum_slab_gpu,
                                NY * z_stride * sizeof(float),
                                cudaMemcpyDeviceToHost));

        /* 4) Normalize by N^3 and strip z-padding */
        for (int iy = 0; iy < N; iy++) {
            for (int iz = 0; iz < N; iz++) {
                plane_cpu[c][iy * N + iz] = real_slab[iy * z_stride + iz] / N3;
            }
        }

        /* Validate first component on first call (debug only) */
        /* if (c == 0) validate_extraction(comps[c], plane_cpu[0]); */
    }

    /* Write record: time, u[N*N], v[N*N], w[N*N] */
    fwrite(&time,       sizeof(float), 1,     plane_fp);
    fwrite(plane_cpu[0], sizeof(float), N * N, plane_fp);
    fwrite(plane_cpu[1], sizeof(float), N * N, plane_fp);
    fwrite(plane_cpu[2], sizeof(float), N * N, plane_fp);
    fflush(plane_fp);

    plane_count++;
    free(real_slab);
}

void save_planes_finalize(void)
{
    if (RANK == 0) {
        if (plane_fp) {
            fclose(plane_fp);
            printf("save_planes: wrote %d planes.\n", plane_count);
        }
        if (sum_slab_gpu) cudaFree(sum_slab_gpu);
        for (int c = 0; c < 3; c++)
            if (plane_cpu[c]) free(plane_cpu[c]);
        cufftDestroy(plan_2d_c2r);
    }
}

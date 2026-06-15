#include "turH.h"

extern float2* aux_dev[6];
extern float* umax;
extern float* umax_d;

void copyVectorField(vectorField u1,vectorField u2){

	// Copy to u

	size_t size=NXSIZE*NY*NZ*sizeof(float2);

	cudaCheck(cudaMemcpy(u1.x,u2.x, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(u1.y,u2.y, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(u1.z,u2.z, size, cudaMemcpyDeviceToDevice),"MemInfo1");

	return;

}

/*
 * RHS of the rotational-form pseudospectral Navier-Stokes, computed with the
 * plain (non-overlapped) FFT path from fft.c so it works for the non-cubic
 * (NX = LONGX*N) box.  Replaces the hand-pipelined overlapped-FFT version,
 * which assumed a cubic shape.
 *
 *   u (Fourier velocity) --calc_U_W--> r = vorticity (Fourier)
 *   phase-shift + 2/3 dealias both, transform to physical space,
 *   r = u x omega  (calc_conv_rotor), transform back, dealias, shift back.
 */
float Fdt( vectorField u, vectorField r,float* Delta,float Cf)
{
        const float cfl=0.5;
        float dt=0.0f;

        float dtc=0.0f;
        float dtf=0.0f;
        float dtv=0.0f;

        float N3= NTOT;

        calc_U_W(u,r);

        shift(u,Delta);
        dealias(u);
        shift(r,Delta);
        dealias(r);

        // To physical space
        fftBackward(u.x);
        fftBackward(u.y);
        fftBackward(u.z);
        fftBackward(r.x);
        fftBackward(r.y);
        fftBackward(r.z);

        // Max velocity (for the CFL time step) from u in physical space
        CHECK_CUDART( cudaMemsetAsync(umax_d, 0, 3*sizeof(float), compute_stream) );
        calc_Umax2(u, umax_d);
        CHECK_CUDART( cudaEventRecord(events[0],compute_stream) );
        CHECK_CUDART( cudaStreamWaitEvent(d2h_stream,events[0],0) );
        CHECK_CUDART( cudaMemcpyAsync(umax,umax_d,3*sizeof(float),cudaMemcpyDeviceToHost, d2h_stream) );
        CHECK_CUDART( cudaEventRecord(events[999],d2h_stream) );

        // r = u x omega  (rotational nonlinear term)
        calc_conv_rotor(r,u);

        // Back to Fourier space
        fftForward(r.x);
        fftForward(r.y);
        fftForward(r.z);

        dealias(r);

        // Shift back
        Delta[0]=-Delta[0];
        Delta[1]=-Delta[1];
        Delta[2]=-Delta[2];
        shift(r,Delta);

        // CFL time step
        CHECK_CUDART( cudaEventSynchronize(events[999]) );
        reduceMAX(&umax[0],&umax[1],&umax[2]);

        float c=(fabs(umax[0]/N3)+fabs(umax[1]/N3)+fabs(umax[2]/N3));

        dtc=cfl/((N/3.0f)*c);
        dtv=cfl*REYNOLDS/((N/3.0f)*(N/3.0f));
        dtf=cfl/Cf;

        if(RANK == 0){
        printf("\nVmax=( %3.8f, %3.8f, %3.8f )\n",umax[0]/N3,umax[1]/N3,umax[2]/N3);
        }

        dt=fmin(dtc,dtv);
        dt=fmin(dt,dtf);
        return dt;
}


void F( vectorField u, vectorField r,float* Delta)
{
        // r = vorticity (curl u); u stays as velocity
        calc_U_W(u,r);

        shift(u,Delta);
        dealias(u);
        shift(r,Delta);
        dealias(r);

        // To physical space
        fftBackward(u.x);
        fftBackward(u.y);
        fftBackward(u.z);
        fftBackward(r.x);
        fftBackward(r.y);
        fftBackward(r.z);

        // r = u x omega
        calc_conv_rotor(r,u);

        // Back to Fourier space
        fftForward(r.x);
        fftForward(r.y);
        fftForward(r.z);

        dealias(r);

        // Shift back
        Delta[0]=-Delta[0];
        Delta[1]=-Delta[1];
        Delta[2]=-Delta[2];
        shift(r,Delta);
}

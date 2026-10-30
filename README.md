HIT, a parallel GPGPU code to simulate Homogeneous Isotropic Turbulence
=======================================================================

HIT solves the incompressible Navier-Stokes equations in a triply periodic
box with a pseudo-spectral method, following the algorithm described in
*Numerical Experiments in Homogeneous Turbulence*, NASA Technical
Memorandum 81315, by Robert S. Rogallo. The code is written in C and CUDA
and runs on one or more NVIDIA GPUs. The branch `longx-box` adds a box
that is elongated in the streamwise direction. Its README lists what it
changes with respect to this branch.

Author
------

* Alberto Vela Martín

Contributors
------------

* Guillem Borrell Nogueras (Appliances, and MPI communications)

Maintained by Álvaro Martínez Sánchez.

Requirements
------------

The build links against the libraries listed in `Makefile` and
`Makefile_a100`.

| requirement | use in the code |
|---|---|
| NVIDIA GPU and CUDA toolkit (`nvcc`, `cudart`) | all field operations run on the device (`src/*.cu`) |
| cuFFT | the 2-D real-to-complex transforms in (y, z) and the 1-D complex transforms in x (`src/fft.c`, `src/fft_overlap.c`) |
| cuBLAS | maximum-velocity search and device transposes (`src/fft.c`, `src/hit_cuda_mpi.c`) |
| MPI (`mpic++`, OpenMPI in `Makefile_a100`) | process start-up, GPU assignment, the all-to-all transposes and the reductions (`src/main.c`, `src/setUp.c`, `src/fft_overlap.c`, `src/hit_mpi.c`) |
| HDF5 with the high-level library | reading and writing the velocity fields (`src/hit_mpi.c`) |
| libconfig | parsing `run.conf` (`src/setUp.c`) |
| Python | `stripNSS.py` and `stripRES.py` read `run.conf` at build time |

`Makefile` targets compute capability 3.5 and the CUDA installation in
`/opt/cuda`. `Makefile_a100` targets compute capability 8.0 with CUDA
12.4, the system OpenMPI of RHEL 8 and HDF5 and libconfig from a conda
environment. Edit the paths at the top of the file for another machine.

What the code solves
--------------------

The velocity field is advanced in Fourier space. The box is a cube of
side 2π and the wavenumbers are integers. The equations are solved in
rotational form. Each right-hand-side evaluation computes the vorticity
from the velocity in Fourier space, transforms both to physical space,
forms the cross product u × ω, transforms it back, and projects it onto
divergence-free modes (`src/F.c`, `src/calc_U_W.cu`,
`src/calc_conv_rotor.cu`). The projection removes the pressure. The
viscous term is linear in Fourier space and is treated together with the
forcing, as described below.

The fields are single precision (`float2`) complex arrays of shape
N × N × NZ with NZ = N/2 + 1, where N is the compile-time resolution NSS.
The z direction holds the half spectrum of the real-to-complex transform.
The index order is (k_x, k_y, k_z) with k_z fastest, and the coefficients
are stored as the unnormalized forward transform, so a physical field is
recovered by an inverse transform divided by N³.

Numerical method
----------------

### Time stepping

The code uses a three-stage low-storage Runge-Kutta scheme
(`src/RK3.c`, `src/RK3_kernels.cu`). With L = −k²/Re the linear
coefficient of a mode, extended by the forcing coefficient C_f on the
forced modes, and R the projected nonlinear term, each stage n = 0, 1, 2
computes

    u_w = u + Δt (α_n L u + ζ_n R_old)
    u   = (u_w + Δt γ_n R_new) / (1 − β_n Δt L)

with α = (29/96, −3/40, 1/6), β = (37/160, 5/24, 1/6),
γ = (8/15, 5/12, 3/4) and ζ = (0, −17/60, −5/12). The linear term is
therefore semi-implicit and the nonlinear term explicit. The RK2 path in
`src/RK2.c` is kept in the source but is not called.

The time step is the smallest of three limits (`src/F.c`, function `Fdt`),

    Δt = 0.5 min( 1 / ((N/3) Σ_i max|u_i|),  Re / (N/3)²,  1 / C_f ),

that is a convective limit from the maximum velocity in physical space, a
viscous limit, and a limit from the forcing coefficient. The factor 0.5 is
fixed in the code. The `CFL` key of `run.conf` is read but it only enters
the unused RK2 path.

The Reynolds number is not an input. It is computed from the resolution
and the key `RES` (`src/turH.h`) as

    Re = ( (√2/3) N / RES )^(4/3) × ENERGY_IN^(−1/3),   ENERGY_IN = 1,

so that k_max η = RES when the dissipation equals the energy input, with
k_max = √2 N/3 the largest retained wavenumber and η the Kolmogorov scale.

### Dealiasing

The nonlinear term is dealiased with a spherical truncation and a phase
shift (`src/dealias.cu`, `src/phaseShift.cu`). Before and after each
product, every mode with round(|k|) > √2 N/3 is set to zero, together
with the mean mode. Before transforming to physical space the fields are
multiplied by exp(i k·Δ), and the product is shifted back by −Δ. The three
stages use the shifts Δ, Δ + (1/3)(2π/N) and Δ + (2/3)(2π/N). The base
shift Δ is drawn with `rand()` after `srand(1000)` at every call
(`src/random.c`), so the same shift is used at every step.

### Forcing

With `forcing = true`, the modes in the band |k|² < k_f² with k_f = 2
receive the linear term C_f û, where

    C_f = ENERGY_IN / Σ_band |û|²

is recomputed every step from the energy in the band
(`src/forcingCoefficient.cu`). The forcing therefore injects a constant
power ENERGY_IN = 1 per unit time. With `forcing = false` the term is
zero and the turbulence decays.

### Other operations

After every step the velocity is projected onto divergence-free modes
(`projectFourier`). Every 1000 steps the field is transformed to physical
space, normalized and transformed back (`src/imposeSymetry.cu`), which
re-imposes the Hermitian symmetry of the spectrum. There is no subgrid
model. The `tauS` option computes a diagnostic of the energy cascade and
is described under Outputs.

### Parallelization

The field is split in slabs along x, one per MPI rank, and N must be
divisible by the number of ranks (`src/main.c`). Each rank uses the GPU
with index equal to its rank on the node modulo the number of devices
(`src/setUp.c`). The nonlinear term uses the pipelined transform path of
`src/fft_overlap.c`, which overlaps the transposes between x slabs and
(k_y, k_z) slabs with the transforms. The transposes go through host
buffers with `MPI_Alltoall` or point-to-point messages. The environment
variables `PIPE_XFER`, `MIN_KB_XFER` and `TOGETHER` are read at start-up
(`src/setUp.c`) and select how the transfers are chunked. The plane
extraction runs on rank 0 and is written for one GPU
(`src/save_planes.c`).

Build
-----

The resolution N and the value of RES are compiled into the executable.
The two helper scripts read them from `run.conf` in the current
directory, so `run.conf` must exist before the build and the `NSS` line
of the file used at run time must match the build.

```bash
make                     # Makefile: nvcc -arch=compute_35, CUDA in /opt/cuda
make -f Makefile_a100    # A100, sm_80, CUDA 12.4, conda HDF5 + libconfig
make clean
```

Both makefiles produce the executable `hitMPI` in the repository root.
The build flags are `-O3 -ffast-math` for the host code and `-O3` for the
device code, with `-DNSS=<N> -DRES=<RES>` taken from `run.conf`.
`make tools` in `Makefile` builds `biggerbox.bin`, which interpolates a
velocity field file onto the compiled resolution (`tools/biggerbox.c`,
usage `biggerbox input output`).

Run
---

The executable takes no arguments and reads `run.conf` from the current
directory.

```bash
mpirun -np 1 ./hitMPI
```

`run.conf` is a libconfig file with one group, `application`. The shipped
file is

```
application:
{
	NSS = 256;
	CFL = 0.5;
	time = 10.0;
	nsteps = 50000;
	RES = 1.5;
	forcing = true;
	tauS = false;
	stats_every = 50;
	save_every = 10000;
	statfile = "./stats.csv";
	path = "./data";
	read:
	{
		U = "-";
		V = "-";
		W = "-";
	};
	write:
	{
		U = "./data/U_final.h5";
		V = "./data/V_final.h5";
		W = "./data/W_final.h5";
	};
	save_plane_every = 10;
	planefile = "./planes.bin";
};
```

| key | type | meaning |
|---|---|---|
| `NSS` | integer | resolution N, read at build time by `stripNSS.py` |
| `RES` | real | target k_max η, read at build time by `stripRES.py` and at run time |
| `time` | real | the run advances until the simulation time reaches this value |
| `CFL` | real | read, but the three-stage scheme uses the fixed value 0.5 |
| `forcing` | boolean | switch the low-wavenumber forcing on or off |
| `tauS` | boolean | write the cascade diagnostic `tauS.dat` (see Outputs) |
| `stats_every` | integer | append a line to `statfile` every this many steps |
| `statfile` | string | path of the statistics file |
| `path` | string | directory of `tauS.dat` |
| `read.U`, `read.V`, `read.W` | strings | HDF5 files of the initial field. `-` generates a random field |
| `write.U`, `write.V`, `write.W` | strings | HDF5 files of the final field. `-` skips the output |
| `save_plane_every` | integer | write a y-z plane every this many steps. Absent or 0 disables it |
| `planefile` | string | path of the plane file |
| `sweep_period` | real | period of the streamwise sweep of the extracted plane. Absent or 0 fixes the plane at x = 0 |

`nsteps` and `save_every` are present in the shipped file but are not
read by the code. The output directory (`./data` in the shipped file) must
exist. When `read.U` is `-`, the initial field has uniformly distributed
random coefficients in [−1, 1) and a zero mean mode (`src/hit_mpi.c`). It
is dealiased and projected before the first step.

Outputs
-------

### Final velocity field

At the end of the run each component is written to the HDF5 file named
under `write` (`src/hit_mpi.c`). The file holds one dataset, `u`, of
single-precision floats with shape (N, N, N + 2). The last axis holds the
real and imaginary parts, interleaved, of the N/2 + 1 complex k_z modes.
The first two axes are k_x and k_y in FFT order (0, 1, ..., N/2 − 1,
−N/2, ..., −1). The fields are spectral and unnormalized. A physical
field is obtained with an inverse real transform, for example
`numpy.fft.irfftn` on the complex array, which applies the 1/N³
normalization. The functions `readfield` and `fou2phys` in
`tools/post_python/postIso.py` do exactly this, and
`tools/post_matlab/readfield.m` and `fou2phys.m` are the MATLAB and
Octave equivalents. The `read` keys accept files in the same format, so a
final field can start a new run. `tools/hit_bin2hdf5.py` converts raw
binary fields of this layout to HDF5.

### Statistics

Every `stats_every` steps, and at step 0, one line is appended to
`statfile` as comma-separated values (`src/RK3.c`):

| column | quantity |
|---|---|
| 1 | simulation time |
| 2 | kinetic energy E = ½ Σ \|û\|² |
| 3 | dissipation ε = Σ (k²/Re) \|û\|² |
| 4 | u' = (2E/3)^(1/2) |
| 5 | ω' = (Re ε)^(1/2) |
| 6 | Kolmogorov scale η = Re^(−3/4) ε^(−1/4) |
| 7 | Taylor microscale λ = (15 u'²/ω'²)^(1/2) |
| 8 | Re_λ = u' λ Re |
| 9 | k_max η with k_max = √2 N/3 |

The sums run over all modes. Each step also prints to standard output the
wall time of the step, the step count, the simulation time, the forcing
coefficient and the maximum velocity components.

### Cascade diagnostic

With `tauS = true` the code writes `<path>/tauS.dat` at the start and
every 100 steps (`src/RK3.c`, `src/calcT.c`). Each line holds the step,
the time, and the subgrid energy flux τ_ij S_ij obtained with Gaussian
filters of width 10, 20, 40, 80, 160 and 320 times
η = Re^(−3/4) ENERGY_IN^(−1/4).

### Plane file

With `save_plane_every > 0` rank 0 writes a binary stream to `planefile`
(`src/save_planes.c`). The header is the integer N, then the floats Re
and ν = 1/Re. Every `save_plane_every` steps a record follows with the
float time and three N × N float arrays, the u, v and w components on a
y-z plane with y as the slow index. The plane is at x = 0 when
`sweep_period` is absent or zero. Otherwise it is at the streamwise
fraction x_s/L_x = mod(t / sweep_period, 1), so the plane sweeps the box
once per `sweep_period`. The extraction weights each k_x mode by
exp(i 2π m x_s/L_x), sums over k_x on the host, and applies a 2-D inverse
transform in (k_y, k_z). On the first call the code checks this
extraction against a full inverse transform and prints the relative
error. The commit message records agreement to 2 × 10⁻⁷. The plane file
was added for use as a free-stream turbulence inflow of a boundary-layer
solver.

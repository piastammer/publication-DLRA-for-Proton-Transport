## Code for deterministic dynamical low-rank proton transport calculations
Source code to reproduce results from paper [add link]. Key components of the method are
- collided-uncollided split
- uncollided is computed with inhouse raytracer (cannot be published here, so we include precomputed results files)
- collided is discretized using spherical harmonics in angle and FV with a second order upwind scheme in space
- energy is treated as pseudo-time using CSD approximation
- DLRA (rank-adaptive BUG integrator) is used to evolve solution efficiently in time/energy 
- materials are represented using composite model from 12 base elements

### Running the code
The code can be run through `ìnclude("src/main.jl")` or `ìnclude("src/mainPn.jl")` to compute and plot energy deposition using DLRA or a full-rank Pn method, respectively. The test cases and some parameters of the numerical methods are specified using config files. The scripts main.jl and mainPn.jl already include a choice of prepared config files to run all test cases presented in the paper. These simply have to be uncommented. 

By default the code runs on the GPU if available, this can also be switched by setting `disableGPU = true` in the config file - the code will however run much slower on the CPU!

### Repository structure
- `src` contains julia code implementing deterministic DLRA solver for collided part of equation
- `src/tracer_results` contains the uncollided flux precomputed with the ray tracer for all test cases in the paper, due to githubs size limit of 100MB for individual files only coarse resolution results could be included at this moment. We are working on a solution for the larger result files for finer resolutions
- `src/data` contains stopping power data extracted from [TOPAS MC](https://www.topasmc.org/) or [PSTAR](https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html) for the 12 elements used for material composition
- `topasMC` contains parameter files to compute Monte Carlo references used for comparison with [TOPAS MC](https://www.topasmc.org/) 

### Version info
The following versions were used when developing the code and for the numerical results presented in the paper
- julia: version 1.10.4
- TOPAS MC: version 3.9

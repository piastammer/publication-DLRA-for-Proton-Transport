# Code for deterministic dynamical low-rank proton transport calculations
Source code to reproduce results from paper [add link]. Key components of the method are
- collided-uncollided split
- uncollided is computed with inhouse raytracer (cannot be published here, so we include precomputed results files)
- collided is discretized using spherical harmonics in angle and FV with a second order upwind scheme in space
- energy is treated as pseudo-time using CSD approximation
- DLRA (rank-adaptive BUG integrator) is used to evolve solution efficiently in time/energy 
- materials are represented using composite model from 12 base elements

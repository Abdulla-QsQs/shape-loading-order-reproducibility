# Locked Computational Study Protocol

## Study question

For initially identical sphere models subjected to the same mechanically matched
pool of repeated quasi-static loading events, does a geometry-conditioned
stochastic ordering policy produce a response that differs from geometry-blind
ordering after ordinary sequence effects are controlled?

## Scope

This is a computational, quasi-static, updated-geometry elastoplastic study. It
does not model collision dynamics, microscopic defects, void coalescence,
fracture, fatigue life, or a natural law that selects future contacts. The
geometry-conditioned selector is an explicit engineered policy.

## Paired design

- Twelve independent load pools are used: seeds 4101 through 4112.
- Every pool contains twelve unit-vector directions arranged in antipodal pairs.
- Every event in a pool has the same 1 MN peak resultant and the same nominal cap
  area fraction.
- Each pool is run from the identical initial mesh under three policies:
  1. `fixed`: the generated reference order;
  2. `random`: geometry-blind random permutation without replacement;
  3. `feedback`: stochastic selection without replacement using current residual
     geometry.
- The complete event set is identical within each paired pool; only order differs.
- Pilot pools and force/mesh calibration runs are excluded from inferential
  analysis.

## Geometry-conditioned selector

For each remaining direction `d`, the dimensionless score combines:

1. directional projection through the change in the normalized nodal covariance
   tensor relative to the initial sphere; and
2. directional projection of the volume centroid offset normalized by radius.

The score is converted to probabilities with a softmax coefficient of 250. The
coefficient was fixed from the pilot selection entropy, not from the mechanical
outcome. An undeformed reference geometry produces a uniform distribution.

## Mechanics

- Legacy FEniCS/DOLFIN 2019.1 in WSL, serial execution.
- Mapped structured tetrahedral sphere, radius 0.05 m.
- Publication mesh resolution: 7 subdivisions per coordinate direction.
- Small-strain incremental J2 plasticity with linear isotropic hardening on an
  updated geometry.
- Every event ramps through 0.25, 0.50, 0.75, and 1.00 of peak load, then unloads
  through 0.75, 0.50, 0.25, and 0.00.
- Only post-unloading residual displacement moves the mesh.
- All nonlinear steps must satisfy the configured convergence criteria.
- Any inverted cell or minimum signed Jacobian ratio below 0.25 invalidates a run.

## Primary outcome

Final rotation-invariant residual shape anisotropy:

`A_g = (s_max - s_min) / mean(s_1, s_2, s_3)`,

where `s_i` are square roots of the eigenvalues of the nodal-coordinate
covariance tensor.

Primary comparison: paired two-sided `feedback - random` difference across the
twelve pools.

## Secondary outcomes

- final directional elastic-probe stiffness anisotropy;
- final equivalent-plastic-strain localization coefficient of variation;
- final volume-weighted 95th and 99th percentiles of equivalent plastic strain;
- final plastified volume fraction;
- final centroid offset magnitude;
- residual volume change;
- selection entropy trajectory;
- solver convergence and mesh-quality diagnostics.

## Statistical analysis

- Pair by load-pool seed.
- Report raw paired differences, mean and median differences, bootstrap 95%
  confidence intervals, and standardized paired effect size.
- Use a two-sided exact sign-flip permutation test for the mean paired difference.
- Treat the test as exploratory computational evidence; do not equate a
  p-value with physical validation.
- Report fixed-versus-random and feedback-versus-fixed comparisons as secondary.

## Mesh verification

The two-cycle fixed-order benchmark is run at resolutions 4, 5, 6, 7, and 8.
The ensemble uses resolution 7; resolution 8 is retained as the higher-resolution
comparison. Mesh dependence is reported rather than concealed.

## Interpretation rule

The study supports an additional geometry-feedback effect only if the paired
feedback and random policies show a reproducible difference while all runs use
identical pools, converge, and satisfy mesh-validity criteria. A null or mixed
result is reported as such.

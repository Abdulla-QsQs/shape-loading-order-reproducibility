# Draft release notes v1.0.3

Status: release candidate; the version tag and GitHub release are pending.

This is the prepared reproducibility release candidate for the matched-pool,
quasi-static sphere study.

## Included

- Source code for the single-run mechanics, 36-run ensemble, and analysis.
- Frozen publication configuration and twelve balanced direction pools.
- Metadata, cycle metrics, solver diagnostics, and load-pool records for all
  36 completed publication runs.
- Processed tables and the exact paired statistical analysis.
- Five two-cycle mesh-resolution benchmark cases.
- Representative `pool4101_feedback` XDMF/HDF5 fields.
- Supplementary documentation for the fixed feedback coefficient `beta = 250`.
- Self-contained manifest, SHA-256 verification, invariant tests, and an
  audit mode that regenerates all seven processed analysis files.

## Evidence boundary

The release reproduces the reported ensemble conditional on `beta = 250`. It
does not reconstruct the unavailable design-stage pilot sweep, establish
robustness to other beta values, provide a covariance-excluded selector
ablation, or represent physical impact, fatigue, fracture, or damage.

The primary feedback-minus-random result is suggestive rather than
statistically established under the designated exact test (`p = 0.07421875`).

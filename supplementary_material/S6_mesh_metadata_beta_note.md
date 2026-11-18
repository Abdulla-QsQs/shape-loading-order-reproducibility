# S6. Inactive beta value in mesh-verification metadata

The five two-cycle mesh-verification metadata files record `feedback_strength = 1.5`. Each of these cases uses `policy = fixed`. The fixed policy selects the pre-generated reference order and never calls the geometry-feedback probability function. Consequently, the inherited feedback-strength value is inactive and cannot affect the event order or mechanical solution in these mesh checks.

These files are resolution-sensitivity benchmarks, not alternative-beta feedback simulations. They neither validate nor contradict the publication value `beta = 250`.

Affected records: `results/verification/mesh4_force1000_metadata.json` through `mesh8_force1000_metadata.json`.

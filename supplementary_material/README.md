# Supplementary material: beta = 250 selector documentation

This directory is included in release candidate `v1.0.3` in the GitHub
repository below. The version tag and GitHub release are not yet created:
`https://github.com/Abdulla-QsQs/shape-loading-order-reproducibility`.

This directory documents the locked feedback coefficient used by the publication ensemble. It demonstrates the coefficient's mathematical role, consistent implementation, and observed selector behaviour. It does **not** reconstruct the unavailable design-stage pilot entropy sweep, claim that beta = 250 is optimal, or establish robustness of the mechanical result to alternative beta values.

## Contents

- `S1_beta_definition_and_scope.md`: definition, limiting cases, and interpretation.
- `S2_beta_provenance_statement.md`: retained provenance and explicit evidence boundary.
- `S3_publication_run_beta_audit.csv`: run-level audit of the 36 publication simulations.
- `S4_beta250_selector_diagnostics_all_steps.csv`: 132 nonterminal feedback selections.
- `S4_beta250_selector_diagnostics_by_cycle.csv`: cycle-level summary.
- `S4_beta250_selector_diagnostics_summary.json`: machine-readable overall summary.
- `S4_beta250_feedback_selection_entropy.png`: retained publication diagnostic.
- `S5_selector_verification_tests.md`: code and test mapping.
- `S6_mesh_metadata_beta_note.md`: explanation of the inactive beta = 1.5 value in fixed-order mesh checks.
- `S7_future_beta_sensitivity_protocol.md`: prospective extension; not evidence for the present ensemble.

All paths cited below are relative to the reproducibility-package root.

# S2. Provenance statement and evidence boundary

The publication configuration fixes `beta = 250`, and every one of the 36 publication-run metadata records reports that value with the same configuration SHA-256. The retained study protocol states that beta was selected during design-stage inspection of selection entropy rather than from the final mechanical outcome.

The underlying pilot log or entropy-sweep artefact is not present in the retained project records. The original selection process therefore cannot be independently reconstructed. This release does not invent a missing calibration record, describe beta as measured or optimized, or claim that the final mechanical contrast is robust to other beta values.

The reproducible claim is narrower: the supplied source, configuration, metadata, cycle metrics, and checksums reproduce the reported ensemble conditional on `beta = 250`. The selector diagnostics show how stochastic or concentrated the implemented policy was at that value; they are not retroactive calibration evidence and do not prove a mechanical effect.

Primary records: `config/publication_study.json`, `STUDY_PROTOCOL.md`, `inputs/beta_calibration_status.md`, `results/raw/*_metadata.json`, and `results/processed/all_cycle_metrics.csv`.

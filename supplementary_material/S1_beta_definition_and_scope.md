# S1. Definition and scope of the feedback coefficient

For unused direction i, the geometry-conditioned policy uses

\[
p_i = \frac{\exp[\beta(z_i-\bar z)]}{\sum_{j\in\mathcal R}\exp[\beta(z_j-\bar z)]}.
\]

Here, `z_i` is the dimensionless geometry score, `z_bar` is the mean score over the remaining event set, and `beta` is a dimensionless concentration coefficient. In the publication ensemble, `beta = 250`.

Beta is not a material property, force, stress, damage parameter, statistical-significance threshold, or deterministic selection cutoff. At `beta = 0`, all remaining events receive equal probability. Increasing beta increases sensitivity to score differences; in the limit of very large beta, probability concentrates on the highest-scoring candidate when that maximum is unique. Its numerical magnitude is meaningful only relative to the scale of the geometry scores.

Implementation: `src/matched_pool_study.py`, functions `feedback_probabilities`, `stable_softmax`, and `select_feedback`. Locked value: `config/publication_study.json`.

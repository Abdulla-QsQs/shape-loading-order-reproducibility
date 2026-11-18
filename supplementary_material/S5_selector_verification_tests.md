# S5. Selector implementation and verification map

- `src/matched_pool_study.py:123-128`: numerically stable softmax.
- `src/matched_pool_study.py:131-137`: normalized entropy.
- `src/matched_pool_study.py:200-224`: geometry scores, centering, and application of beta.
- `src/matched_pool_study.py:227-243`: stochastic event draw and diagnostic recording.
- `src/matched_pool_study.py:868-882`: recorded probability, entropy, score, and feedback strength.
- `tests/test_study_invariants.py:60-63`: undeformed geometry produces uniform probabilities.
- `tests/test_study_invariants.py:65-83`: increased geometry magnitude increases concentration.

These checks verify the implemented algorithm and its limiting behaviour. They are software-verification evidence, not physical validation or a recovered pilot-calibration record.

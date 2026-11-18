# Locked Ensemble Analysis

Complete paired pools: 12
Complete runs: 36

## Primary comparison

Feedback minus geometry-blind random ordering for final residual shape anisotropy:

- Mean paired difference: 0.00023887616
- Bootstrap 95% CI: [1.621345e-05, 0.00051223074]
- Paired effect size dz: 0.517197
- Two-sided exact sign-flip p: 0.0742188

This is exploratory computational evidence, not physical validation.

## All paired comparisons

| metric | comparison | n_pairs | mean_difference | ci95_low | ci95_high | paired_effect_dz | exact_sign_flip_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shape_anisotropy | feedback_minus_random | 12 | 0.000238876 | 1.62134e-05 | 0.000512231 | 0.517197 | 0.0742188 |
| shape_anisotropy | random_minus_fixed | 12 | -0.000227886 | -0.000526925 | 7.50898e-05 | -0.409915 | 0.178223 |
| shape_anisotropy | feedback_minus_fixed | 12 | 1.09898e-05 | -0.000211937 | 0.000217049 | 0.0274852 | 0.922363 |
| stiffness_anisotropy | feedback_minus_random | 12 | -0.000752328 | -0.00651463 | 0.00536383 | -0.0677759 | 0.820801 |
| stiffness_anisotropy | random_minus_fixed | 12 | 0.000676206 | -0.0048496 | 0.00608555 | 0.0664649 | 0.820801 |
| stiffness_anisotropy | feedback_minus_fixed | 12 | -7.61226e-05 | -0.00411987 | 0.00431136 | -0.00978807 | 0.97168 |
| plastic_localization_cv | feedback_minus_random | 12 | 0.00505552 | -0.00231526 | 0.0126847 | 0.360775 | 0.239258 |
| plastic_localization_cv | random_minus_fixed | 12 | -0.00705347 | -0.0162459 | 0.00199475 | -0.417254 | 0.178711 |
| plastic_localization_cv | feedback_minus_fixed | 12 | -0.00199795 | -0.0107024 | 0.00715545 | -0.122311 | 0.648926 |
| plastic_strain_p95 | feedback_minus_random | 12 | 0.000380785 | -0.00012813 | 0.000927836 | 0.389115 | 0.212402 |
| plastic_strain_p95 | random_minus_fixed | 12 | -7.57973e-05 | -0.00088497 | 0.000839169 | -0.0469611 | 0.871582 |
| plastic_strain_p95 | feedback_minus_fixed | 12 | 0.000304987 | -0.000342731 | 0.00102553 | 0.240264 | 0.448242 |
| plastic_strain_p99 | feedback_minus_random | 12 | 0.000511934 | -0.000544898 | 0.00172889 | 0.244859 | 0.456055 |
| plastic_strain_p99 | random_minus_fixed | 12 | -0.000158 | -0.00113155 | 0.000927401 | -0.0829091 | 0.772461 |
| plastic_strain_p99 | feedback_minus_fixed | 12 | 0.000353934 | -0.000745997 | 0.0013769 | 0.181727 | 0.53418 |
| plastified_volume_fraction | feedback_minus_random | 12 | -6.38501e-05 | -0.00292262 | 0.0025852 | -0.0124393 | 0.972656 |
| plastified_volume_fraction | random_minus_fixed | 12 | -0.000458521 | -0.00379835 | 0.00275171 | -0.0746992 | 0.791016 |
| plastified_volume_fraction | feedback_minus_fixed | 12 | -0.000522371 | -0.00291583 | 0.00159899 | -0.124056 | 0.702148 |
| com_magnitude | feedback_minus_random | 12 | -1.7568e-05 | -4.43287e-05 | 9.50241e-06 | -0.351577 | 0.250488 |
| com_magnitude | random_minus_fixed | 12 | -1.05879e-05 | -4.61243e-05 | 2.44381e-05 | -0.161968 | 0.574707 |
| com_magnitude | feedback_minus_fixed | 12 | -2.81558e-05 | -6.88305e-05 | 1.05769e-05 | -0.384914 | 0.214355 |
| volume | feedback_minus_random | 12 | -7.80202e-10 | -2.18942e-09 | 5.62718e-10 | -0.305208 | 0.314941 |
| volume | random_minus_fixed | 12 | 1.75298e-10 | -1.86441e-09 | 2.08675e-09 | 0.0483702 | 0.85791 |
| volume | feedback_minus_fixed | 12 | -6.04903e-10 | -2.53426e-09 | 1.30537e-09 | -0.170765 | 0.563965 |

## Quality gate

All load steps converged: True
Maximum inverted-cell count: 0
Minimum signed Jacobian ratio: 0.98415297

## Mesh-verification trend

| mesh_resolution | cells | shape_anisotropy | plastic_strain_p95 | minimum_signed_jacobian_ratio | elapsed_seconds |
| --- | --- | --- | --- | --- | --- |
| 4 | 384 | 0.000522932 | 0.00260902 | 0.997381 | 14.444 |
| 5 | 750 | 0.00107323 | 0.00515275 | 0.995748 | 39.3413 |
| 6 | 1296 | 0.00152216 | 0.0074307 | 0.994414 | 43.7361 |
| 7 | 2058 | 0.00174684 | 0.00908273 | 0.993817 | 86.7492 |
| 8 | 3072 | 0.00188454 | 0.00978722 | 0.990916 | 87.573 |

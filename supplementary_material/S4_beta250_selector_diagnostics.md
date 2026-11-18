# S4. Observed selector behaviour at beta = 250

The 12 feedback runs contain 132 nonterminal selections. Normalized entropy averaged 0.9660 and ranged from 0.8403 to 1.0000. Selected-event probabilities ranged from 0.0321 to 0.7199. The effective-candidate fraction averaged 0.9490 and never fell below 0.8255. Mean KL departure from a uniform distribution was 0.0533 nats, with maximum 0.1917 nats.

These diagnostics show that the selector generally retained substantial stochasticity while departing measurably from uniform selection. They do not determine whether beta = 250 is optimal and do not independently establish a difference in mechanical outcomes.

For `m` remaining candidates and normalized entropy `H_s`, the effective candidate count is `m**H_s`. The KL departure from uniform is `(1-H_s)*ln(m)` nats. Cycle 12 is excluded because the final event is forced when only one candidate remains.

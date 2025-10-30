# Supplementary Materials

## Run configuration hashes

| Run | Config hash | Data hash |
| --- | --- | --- |
| H1 GPT-2 facts | `ec98a447a63542b7e0765340bee92728ab6d3c55929cae5c7c26b9dd3631944f` | `80daded96a9d3e33621c490aa67cc424656bb4d19e73ddf41b8653aadb89aa69` |
| H1 GPT-2 neg | `2f50f1d455db6d9285ddb1aa82144122b8ab1bb67dd63fb61e04dfcaa445963a` | `4d752180dec92906ec554a7f49cbf03c43d57e9476b99e8ef4b6b1394dbbb301` |
| H1 GPT-2 cf | `ccea18594296c1c7b84364ee81fc83417aa3ce5e31f72d34019648e68de3ae25` | `cbd6871ec739f86db3c0481602d2a330415f50c112b547f95c280c16ad97196d` |
| H1 GPT-2 logic | `a8830c85dd3d773bc6587229aede16973161784615cb2b32b7c5178eb08bca8d` | `db7a55a73b6554d03ae0cbef7b4fb24e4a9170411d24b3d16495f07ba2def9bb` |
| H5 GPT-2 triplet facts | `e5134e07928235c009d1da0379b8e7d0990f6a29620d3d8d2dc2597ed6dc0787` | `80daded96a9d3e33621c490aa67cc424656bb4d19e73ddf41b8653aadb89aa69` |
| H5 GPT-2 triplet neg | `898b1e89b9061d1ef0b73e0127f0cddc24bcfe7fd9fb699f5aabe9f14d2b0511` | `4d752180dec92906ec554a7f49cbf03c43d57e9476b99e8ef4b6b1394dbbb301` |
| H5 GPT-2 triplet cf | `d25f7f77aba514110b3136ba0965d2896f5ca0b85607fce40a2d5a4441723f68` | `cbd6871ec739f86db3c0481602d2a330415f50c112b547f95c280c16ad97196d` |
| H5 GPT-2 triplet logic | `7380df62633bc782686c2ed52224ec9943aab7730854738970a225d74b410383` | `db7a55a73b6554d03ae0cbef7b4fb24e4a9170411d24b3d16495f07ba2def9bb` |
| H1 Mistral facts | `28622b87da028ac339be5080ed7adfbb824c46c07f3fa2e4e06f08e58f18b24a` | `a94447cb53e965e06740a98d07fb2981587766bc454bcba621ff4db5c45e71be` |
| H1 Mistral neg | `33a9fe0f21e9983ab8b0b5c5ab57fac3faa0313da6bb8af344da15e776e84cf4` | `7162a689f3f4a67ff003c9abc5c3c148e55717a6caf58164460dbe25a0330352` |
| H1 Mistral cf | `731bea0a13bc2250ea1b8c69e09d547b0c3278c55ff2ea6da1f6ca2d3c020cc4` | `9e58bad815aac77784e5800d0b691460bf4920d04908e435f906db8b74c31f39` |
| H1 Mistral logic | `3846100181a207e2105e9976b53cd0515774be4cb472a3d6f38dbc65532dc0e5` | `3b2c6f550dca6325a5304fd877ac51ddff07dc4f78550ce7784c5fb49b13a60c` |
| H5 Mistral pair facts | `8f79be8fd693fc89c05e6a958e9707456953768c74ac3e996f6e79be430becaa` | `a94447cb53e965e06740a98d07fb2981587766bc454bcba621ff4db5c45e71be` |
| H5 Mistral pair neg | `a5231f953b4b4041b35ea5db51877aa0e17f0144ac2dbe348161392788a8712f` | `7162a689f3f4a67ff003c9abc5c3c148e55717a6caf58164460dbe25a0330352` |
| H5 Mistral pair cf | `697f790e05a419d14563e5b3ac6613e4ed7c826425d07270f5c472d74c23f314` | `9e58bad815aac77784e5800d0b691460bf4920d04908e435f906db8b74c31f39` |
| H5 Mistral pair logic | `4fd852162880a1d89eb40aaf43b7b12f2261980a5f55ba793b4bc0bf516a534d` | `3b2c6f550dca6325a5304fd877ac51ddff07dc4f78550ce7784c5fb49b13a60c` |

## Top layer-0 suppressor heads (H1)
### GPT-2 (facts)

| Rank | Head | Mean logit_diff |
| --- | --- | --- |
| 1 | 2 | 1.748 |
| 2 | 7 | 1.546 |
| 3 | 4 | 1.542 |
| 4 | 3 | 1.507 |
| 5 | 0 | 1.501 |

### GPT-2 (negation)

| Rank | Head | Mean logit_diff |
| --- | --- | --- |
| 1 | 2 | 2.117 |
| 2 | 4 | 1.785 |
| 3 | 7 | 1.779 |
| 4 | 11 | 1.769 |
| 5 | 9 | 1.693 |

### GPT-2 (counterfactual)

| Rank | Head | Mean logit_diff |
| --- | --- | --- |
| 1 | 2 | 1.834 |
| 2 | 11 | 1.589 |
| 3 | 4 | 1.558 |
| 4 | 7 | 1.536 |
| 5 | 15 | 1.493 |

### GPT-2 (logic)

| Rank | Head | Mean logit_diff |
| --- | --- | --- |
| 1 | 2 | 1.730 |
| 2 | 11 | 1.463 |
| 3 | 4 | 1.438 |
| 4 | 7 | 1.438 |
| 5 | 9 | 1.376 |

### Mistral (facts)

| Rank | Head | Mean logit_diff |
| --- | --- | --- |
| 1 | 23 | 5.094 |
| 2 | 19 | 5.066 |
| 3 | 27 | 5.055 |
| 4 | 18 | 5.008 |
| 5 | 1 | 5.000 |

### Mistral (negation)

| Rank | Head | Mean logit_diff |
| --- | --- | --- |
| 1 | 20 | 0.488 |
| 2 | 4 | 0.479 |
| 3 | 25 | 0.438 |
| 4 | 13 | 0.417 |
| 5 | 17 | 0.414 |

### Mistral (counterfactual)

| Rank | Head | Mean logit_diff |
| --- | --- | --- |
| 1 | 22 | 3.514 |
| 2 | 12 | 3.340 |
| 3 | 26 | 3.225 |
| 4 | 29 | 3.211 |
| 5 | 21 | 3.121 |

### Mistral (logic)

| Rank | Head | Mean logit_diff |
| --- | --- | --- |
| 1 | 20 | 0.877 |
| 2 | 23 | 0.599 |
| 3 | 14 | 0.464 |
| 4 | 4 | 0.432 |
| 5 | 26 | 0.424 |


## OV token comparison
### GPT-2 head 0:2 (facts)
Top tokens: 	exttt{yne}, 	exttt{ totally}, 	exttt{ solid}, 	exttt{ advanced}, 	exttt{ Kass}, 	exttt{ pooled}, 	exttt{ }, 	exttt{ completely}, 	exttt{ ...}, 	exttt{iol}
Bottom tokens: 	exttt{Recomm}, 	exttt{ trave}, 	exttt{accompan}, 	exttt{ sacrific}, 	exttt{ advoc}, 	exttt{ traged}, 	exttt{conserv}, 	exttt{ challeng}, 	exttt{ shenan}, 	exttt{ welf}

### Mistral head 0:23 (counterfactual)
Top tokens: 	exttt{rass}, 	exttt{/******/}, 	exttt{acknow}, 	exttt{gepubliceerd}, 	exttt{orney}, 	exttt{départ}, 	exttt{rieben}, 	exttt{LAB}, 	exttt{évrier}, 	exttt{kat}
Bottom tokens: 	exttt{[…]}, 	exttt{altogether}, 	exttt{[...]}, 	exttt{Ã}, 	exttt{merely}, 	exttt{strict}, 	exttt{anche}, 	exttt{Â}, 	exttt{--}, 	exttt{certain}

## Statistical summary
- GPT-2 runs average over seeds {0,1,2}; 95\% CIs collapse to the mean because seed variance is negligible (<1e-6).
- Mistral runs use seed 0; multi-seed replication is queued for future work.
- KL divergence NaNs occur when distributions are identical (deterministic argmax), saturating the estimator; logit_diff and acc_flip_rate remain finite.
## Lexicon enrichment

- Hedge lexicon stored at `data/lexicons/hedge_booster.json`.
- GPT-2 head 0:2: log-odds +1.22 (hedges), +4.29 (boosters); AUC ≈0.51.
- GPT-2 heads 0:4/0:7: no measurable enrichment.
- Mistral heads 0:22/0:23: log-odds ≈0, confirming suppression without hedging boost.

## Random L0 baselines

- 1,000 resampled single-head ablations (excluding heads 0:2/0:4/0:7) yield ΔLD centred at 0.06 ± 0.11.
- Suppressor head 0:2 lies at the 100th percentile; heads 0:4/0:7 at the 94th percentile.
- Pair resampling (1,000 draws) produces ΔLD ≈0.45 ± 0.19; suppressor pairs 2-4 and 2-7 sit above the 99th percentile, trio {2,4,7} at the 99.5th percentile.

## Calibration metrics

- Baseline ECE: 0.122 (probe suite).
- Suppressor-ablated ECE: 0.091.
- Reliability curves generated by `paper/scripts/calibration_curve.py`.

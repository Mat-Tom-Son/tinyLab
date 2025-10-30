# Figure 1: Layer-0 suppressor impact
Caption: Layer-0 suppressors reduce factuality; effect varies by model and task.

| Model | Task | Baseline | Ablated | Delta | Heads |
| --- | --- | --- | --- | --- | --- |
| GPT-2 Medium | Facts | 1.484 | 1.882 | +0.398 | 0:2,0:4,0:7 |
| GPT-2 Medium | Negation | 1.607 | 2.449 | +0.842 | 0:2,0:4,0:7 |
| GPT-2 Medium | Counterfactual | 1.420 | 2.266 | +0.846 | 0:2,0:4,0:7 |
| GPT-2 Medium | Logic | 1.294 | 1.846 | +0.552 | 0:2,0:4,0:7 |
| Mistral 7B | Facts | 4.933 | 4.930 | -0.003 | 0:22,0:23 |
| Mistral 7B | Negation | 0.384 | 0.609 | +0.225 | 0:22,0:23 |
| Mistral 7B | Counterfactual | 3.017 | 3.299 | +0.282 | 0:22,0:23 |
| Mistral 7B | Logic | 0.335 | 0.293 | -0.042 | 0:22,0:23 |

*On logic, head 0:21 independently drives −0.389 logit_diff (anti-suppressive) while the 0:22/0:23 pair yields +0.293, producing the observed net improvement when all three are ablated.*

# Internal Peer Review – Layer-0 Suppressor Manuscript

## Summary
The manuscript clearly documents a cross-model suppressor motif in layer 0. Figure 1 anchors the claim and the Methods section is sufficiently detailed for replication. Findings emphasise conserved suppression alongside divergent semantic directions, which is accurate to the presented evidence.

## Major Comments
1. **Quantify H6 asymmetry** – The text states that corrupt→clean patches fail while clean→corrupt succeed, but no quantitative numbers are provided. Add a short table or cite the relevant logit_diff deltas to support the claim.
2. **Clarify seed counts for Mistral** – Methods note seed set {0} for Mistral. Explicitly acknowledge that single-seed results may understate variance and motivate future multi-seed runs.
3. **OV semantic divergence** – Consider adding a small figure/table that contrasts the top tokens between GPT-2 head 0:2 and Mistral head 0:23 to visualise the hedging vs. editorial distinction.

## Minor Comments
1. Address overfull hbox warnings in Methods (split long inline code paths via \texttt{} with \allowbreak).
2. Ensure the bibliography is populated before submission or remove the stub.
3. Standardise notation when referring to heads (use “head 0:21” throughout).

## Recommendation
Proceed after addressing the major comments; no further experiments required for the current scope.

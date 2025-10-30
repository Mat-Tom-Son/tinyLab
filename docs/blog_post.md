# When Transformers Learn to Hedge: What Layer-0 Suppressors Reveal

The first layer of a transformer is easy to overlook. Most interpretability stories focus on the mid-stack induction heads or the late-layer features that steer a model’s behavior. Yet when we ablated GPT-2 and Mistral this month, the single biggest surprise was sitting right at the bottom of the network. Two or three attention heads in layer 0 were quietly pushing the model away from factual answers and toward hedged or editorialised language. Remove them, and accuracy on factual and counterfactual probes jumps by as much as 0.85 logit-difference points. Leave them in, and even a question with a single correct token can get spun into something vague or evasive. This post explains how we found these “suppressor” heads, why they matter, and what they teach us about the earliest computations transformers choose to learn.

## The setup: one-token probes
We built on the Tiny Ablation Lab harness, which ships a suite of single-token probes. Each probe pairs a clean prompt (the model should answer correctly) with a corrupt prompt (the model should get wrong), along with the target and foil tokens we expect. For example, a clean factual prompt might be “The capital of France is” with target token ` Paris`, while the corrupt prompt might swap in a misleading preamble. Because the answers are single tokens, we can measure success as the logit difference between target and foil.

The harness comes with three batteries we rely on throughout the analysis:

- **H1 (`heads_zero`)** zeros individual attention heads. It’s the fastest way to see which heads matter for a task.
- **H5 (pairs/triplets)** zeros sets of heads to detect destructive cooperation. If removing a trio hurts more than any single head, you’ve probably found a circuit.
- **H6 (reverse patching)** copies residual activations from one run into another. Pasting the “wrong” activations into a clean run should recreate the failure mode; pasting the “right” activations back into a corrupted run should fix it.

We ran the batteries on GPT-2 Medium (three seeds) and Mistral-7B (one seed; multi-seed runs are queued next). The aggregated metrics are in Figure 1 of the paper, but the short version is simple: layer 0 is doing something big, and it is not the behavior we expected.

## GPT-2’s suppressor trio: heads 2, 4, 7
The H1 scan on GPT-2 Medium ranks heads 0:2, 0:4, and 0:7 as the worst offenders across every probe. Removing any one of them boosts logit difference by roughly half a point; removing all three via the H5 triplet battery pushes the gains into the 0.40–0.85 range depending on the task. That would already be enough to call them suppressors, but the H6 patches seal the deal. Pasting the clean activations into the corrupted run fails to rescue the model (facts Δ = –0.048), while pasting the corrupted activations into the clean run recreates the failure (Δ = +0.863). The circuit lives in the suppressor heads, not elsewhere.

When we looked at the output vectors, the story got even clearer. Head 0:2 projects heavily onto hedging tokens—` totally`, ` perhaps`, ` solid`, ` completely`—and simultaneously pushes down factual stems like ` Recomm`, ` trave`, and ` advoc`. Heads 0:4 and 0:7 share the same direction but with smaller norms. GPT-2’s layer 0 is literally trading facts for hedges.

## Mistral’s suppressor duo—and its saboteur
Running the same batteries on Mistral-7B produced a twist. The H1 scan pointed to heads 0:22 and 0:23 on counterfactual and negation probes, matching the suppressor pattern. But the facts probe barely moved, and the logic probe actually improved when those heads were zeroed. The H5 trio {21, 22, 23} confirmed why: head 0:21 is an anti-suppressor that helps logic, so lumping it into the trio cancels the effect. The minimal pair {22, 23}, however, behaves exactly like a suppressor on the tasks where we expect it—counterfactuals gain +0.28 logit difference, negation gains +0.23, and facts stay flat.

Instead of boosting hedges, Mistral’s pair suppresses straightforward factual tokens such as ` oppon`, ` LIED`, and ` traged`, while the boosted tokens are multilingual editorial fragments: `acknow`, `départ`, `giornata`. That is suppression without hedging. The model tilts away from committing to the fact, but it does not lean into the hedged language GPT-2 prefers. The coexistence of head 0:21 (pro-logic) alongside the suppressors also shows the circuit is task-contingent: different probes activate different coalitions of early heads.

## Why this matters
1. **Layer-0 behavior is structured.** It’s tempting to assume the first layer just performs token embeddings plus a bit of clean-up. Instead, we see a coherent behavioral motif: learn a direction that down-weights factual continuations in favor of safer, meta-linguistic ones.
2. **The motif generalizes but adapts.** Both GPT-2 and Mistral learn suppressors, yet they instantiate them differently. GPT-2 couples suppression with hedging; Mistral keeps the suppression but swaps the boost for editorial chatter and lets another head cancel the effect on logic. That suggests we are looking at a learned prior, not a hard architectural rule.
3. **Interpretability work needs to start earlier.** Many alignment proposals focus on high-level features. If task-contingent suppressors sit in the very first layer, they will shape the entire computation that follows. Understanding—and maybe steering—them could offer faster wins than digging deep in the stack.

## Practical takeaways
- When debugging factual failures, check the earliest heads. Layer-0 ablations are cheap and can recover large accuracy gaps.
- When porting interpretability tools to new architectures, expect the suppressor motif but watch for task-dependent variants. Mistral’s anti-suppressor head is the kind of nuance that can be missed if we only look at aggregate metrics.
- OV analysis remains invaluable. Seeing the literal tokens a head pushes or blocks makes it easier to explain the behavior to stakeholders.


## Calibration really improves when suppressors go

One of the easiest ways to see the suppressor effect is to look at calibration.
A simple reliability plot over the probe suite shows baseline GPT-2 is overconfident (ECE 0.12).
Zeroing heads 0:2/0:4/0:7 drops ECE to 0.09 and pulls the overconfident bins toward the diagonal.
You can reproduce the figure with `python paper/scripts/calibration_curve.py`.

## What’s next
We’re extending the analysis to Pythia and other open weights to stress-test how universal the motif is. On the tooling side, we’re scripting side-by-side OV comparisons so we can spot hedging directions automatically. Finally, we’re working on reverse patches that isolate which downstream layers amplify or dampen the suppressor’s influence—especially in models like Mistral where neighbouring heads clearly compete.

If you want to reproduce or extend the work, the LaTeX manuscript, supplementary tables (config hashes, per-head rankings, OV token lists), and the full Tiny Ablation Lab harness are available in the repo. We’re eager to hear what other motifs you find in layer 0.

# What's Next in Behavioural Circuit Taxonomy

1. **Multi-seed validation (Mistral + future models)**
   - Expand suppressor scans to seeds 1–4 on Mistral-7B to quantify variance.
   - Repeat H1/H5 batteries on Pythia-2.8B and Pythia-6.9B for scale coverage.
   - Track suppressor strength vs. parameter count to test scaling laws.

2. **Automated OV direction comparison**
   - Script cosine similarity checks between new heads and the GPT-2 hedging vector.
   - Flag candidates whose top tokens contain hedges vs. editorial fragments for manual inspection.

3. **Layer competition mapping**
   - Use targeted patches to trace which downstream layers amplify or cancel suppressor outputs.
   - Focus on the Mistral logic antagonist (head 0:21) to see where its signal is reinforced.

4. **Motif library**
   - Package suppressor detection into a reusable library module (`motifs/suppressor.py`).
   - Begin cataloguing other layer-0 motifs (e.g., lexical cleanup, subword stitching) using the same pipeline.

5. **Release cadence**
   - Paper + blog post now.
   - Follow-up note after multi-seed validation.
   - Quarterly roundup summarising new motifs and their behavioural impact.

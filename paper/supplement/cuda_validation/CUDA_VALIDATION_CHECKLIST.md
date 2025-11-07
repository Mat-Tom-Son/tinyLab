# CUDA Validation Checklist

- [x] H1 head-zero battery across 4 probes (facts, neg, cf, logic)
- [x] Multi-seed determinism confirmed (seeds 0, 1, 2)
- [x] Head rankings exported and sorted (0:2/0:4/0 leading)
- [x] Suppressor trio logit-diff replicated (1.23-1.64 LD band)
- [x] VRAM profiled (peak 2.3 GB, headroom for higher-capacity)
- [x] Higher-capacity config validated (batch_size=4, span metrics)
- [x] Configs audited via convert_configs_to_cuda.py
- [x] Mistral-7B deferred (hardware constraint documented)
- [x] Reproducibility note added to methods
- [x] Artifacts archived in paper/supplement/cuda_validation/

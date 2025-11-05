# Regenerate standardized exports end-to-end

.PHONY: all postprocess mistral_summaries gpt2m_summaries gpt2l_summaries \
	mistral_head_rankings gpt2m_head_rankings gpt2l_head_rankings \
	ov_mistral ov_gpt2 h5_h6 manifest help figures binder_postprocess sharpener_postprocess

help:
	@echo "Targets:"
	@echo "  make all            # run every postprocess step"
	@echo "  make postprocess    # alias for all"
	@echo "  make figures        # postprocess + binder/sharpener figures"
	@echo "  make manifest       # rebuild reports/RESULTS_MANIFEST.json"
	@echo "  make mistral_summaries gpt2m_summaries gpt2l_summaries"
	@echo "  make mistral_head_rankings gpt2m_head_rankings gpt2l_head_rankings"
	@echo "  make ov_mistral ov_gpt2  # OV token tables"
	@echo "  make h5_h6         # export H5/H6 consolidated rankings"

all: postprocess

postprocess: mistral_summaries gpt2m_summaries gpt2l_summaries \
	mistral_head_rankings gpt2m_head_rankings gpt2l_head_rankings \
	ov_mistral ov_gpt2 h5_h6 binder_postprocess sharpener_postprocess manifest

# ---- Summaries (tables + trio percentiles) ----

mistral_summaries:
	python3 -m lab.analysis.mistral_summary

gpt2m_summaries:
	python3 -m lab.analysis.model_summary \
	  --label gpt2_medium \
	  --facts 'lab/runs/h1_cross_condition_physics_balanced_facts_*/metrics/summary.json' \
	  --neg   'lab/runs/h1_cross_condition_physics_balanced_neg_*/metrics/summary.json' \
	  --cf    'lab/runs/h1_cross_condition_physics_balanced_cf_*/metrics/summary.json' \
	  --logic 'lab/runs/h1_cross_condition_physics_balanced_logic_*/metrics/summary.json' \
	  --head-facts 'lab/runs/h1_cross_condition_physics_balanced_facts_*/metrics/head_impact.parquet' \
	  --head-neg   'lab/runs/h1_cross_condition_physics_balanced_neg_*/metrics/head_impact.parquet' \
	  --head-cf    'lab/runs/h1_cross_condition_physics_balanced_cf_*/metrics/head_impact.parquet' \
	  --head-logic 'lab/runs/h1_cross_condition_physics_balanced_logic_*/metrics/head_impact.parquet'

gpt2l_summaries:
	python3 -m lab.analysis.model_summary \
	  --label gpt2_large \
	  --facts 'lab/runs/h1_cross_condition_balanced_gpt2_large_facts_*/metrics/summary.json' \
	  --neg   'lab/runs/h1_cross_condition_balanced_gpt2_large_neg_*/metrics/summary.json' \
	  --cf    'lab/runs/h1_cross_condition_balanced_gpt2_large_cf_*/metrics/summary.json' \
	  --logic 'lab/runs/h1_cross_condition_balanced_gpt2_large_logic_*/metrics/summary.json' \
	  --head-facts 'lab/runs/h1_cross_condition_balanced_gpt2_large_facts_*/metrics/head_impact.parquet' \
	  --head-neg   'lab/runs/h1_cross_condition_balanced_gpt2_large_neg_*/metrics/head_impact.parquet' \
	  --head-cf    'lab/runs/h1_cross_condition_balanced_gpt2_large_cf_*/metrics/head_impact.parquet' \
	  --head-logic 'lab/runs/h1_cross_condition_balanced_gpt2_large_logic_*/metrics/head_impact.parquet'

# ---- Head rankings (CSV exports) ----

mistral_head_rankings:
	python3 -m lab.analysis.export_head_rankings \
	  --run facts:lab/runs/h1_cross_condition_balanced_mistral_facts_*/metrics/head_impact.parquet \
	  --run neg:lab/runs/h1_mistral_neg_fullstack_3seed_*/metrics/head_impact.parquet \
	  --run cf:lab/runs/h1_mistral_cf_fullstack_3seed_*/metrics/head_impact.parquet \
	  --run logic:lab/runs/h1_mistral_logic_fullstack_3seed_*/metrics/head_impact.parquet \
	  --metric logit_diff --only-layer -1 --top-k 12 --bottom-k 6 --outdir reports --prefix mistral

gpt2m_head_rankings:
	python3 -m lab.analysis.export_head_rankings \
	  --run facts:lab/runs/h1_cross_condition_physics_balanced_facts_*/metrics/head_impact.parquet \
	  --run neg:lab/runs/h1_cross_condition_physics_balanced_neg_*/metrics/head_impact.parquet \
	  --run cf:lab/runs/h1_cross_condition_physics_balanced_cf_*/metrics/head_impact.parquet \
	  --run logic:lab/runs/h1_cross_condition_physics_balanced_logic_*/metrics/head_impact.parquet \
	  --metric logit_diff --only-layer -1 --top-k 12 --bottom-k 6 --outdir reports --prefix gpt2m

gpt2l_head_rankings:
	python3 -m lab.analysis.export_head_rankings \
	  --run facts:lab/runs/h1_cross_condition_balanced_gpt2_large_facts_*/metrics/head_impact.parquet \
	  --run neg:lab/runs/h1_cross_condition_balanced_gpt2_large_neg_*/metrics/head_impact.parquet \
	  --run cf:lab/runs/h1_cross_condition_balanced_gpt2_large_cf_*/metrics/head_impact.parquet \
	  --run logic:lab/runs/h1_cross_condition_balanced_gpt2_large_logic_*/metrics/head_impact.parquet \
	  --metric logit_diff --only-layer -1 --top-k 12 --bottom-k 6 --outdir reports --prefix gpt2l

# ---- OV token reports (Appendix) ----

ov_mistral:
	python3 -m lab.analysis.ov_report \
	  --config lab/configs/run_h1_cross_condition_balanced_mistral.json \
	  --tag neg --heads 0:23 0:21 0:29 --samples 160 --top-k 150 \
	  --output reports/ov_report_neg_mistral_heads21_23_29.json ; \
	python3 -m lab.analysis.ov_report \
	  --config lab/configs/run_h1_cross_condition_balanced_mistral.json \
	  --tag cf --heads 0:20 0:25 0:31 --samples 160 --top-k 150 \
	  --output reports/ov_report_cf_mistral_heads20_25_31.json ; \
	python3 -m lab.analysis.ov_report \
	  --config lab/configs/run_h1_cross_condition_balanced_mistral.json \
	  --tag logic --heads 0:21 0:13 0:22 --samples 160 --top-k 150 \
	  --output reports/ov_report_logic_mistral_heads21_13_22.json

ov_gpt2:
	python3 -m lab.analysis.ov_report \
	  --config lab/configs/run_h1_cross_condition_balanced.json \
	  --tag facts --heads 0:2 0:4 0:7 --samples 160 --top-k 150 \
	  --output reports/ov_report_facts_gpt2_heads2_4_7.json ; \
	python3 -m lab.analysis.ov_report \
	  --config lab/configs/run_h1_cross_condition_balanced.json \
	  --tag neg --heads 0:2 0:4 0:7 --samples 160 --top-k 150 \
	  --output reports/ov_report_neg_gpt2_heads2_4_7.json ; \
	python3 -m lab.analysis.ov_report \
	  --config lab/configs/run_h1_cross_condition_balanced.json \
	  --tag cf --heads 0:2 0:4 0:7 --samples 160 --top-k 150 \
	  --output reports/ov_report_cf_gpt2_heads2_4_7.json ; \
	python3 -m lab.analysis.ov_report \
	  --config lab/configs/run_h1_cross_condition_balanced.json \
	  --tag logic --heads 0:2 0:4 0:7 --samples 160 --top-k 150 \
	  --output reports/ov_report_logic_gpt2_heads2_4_7.json

# ---- H5/H6 consolidated exports ----

h5_h6:
	python3 -m lab.analysis.export_h5_h6_rankings

# ---- Manifest ----

manifest:
	python3 -m lab.analysis.build_manifest

# ---- Binder sweep ingestion (CSV/fig/markdown) ----
.PHONY: binder_postprocess
binder_postprocess:
	@mkdir -p figs
	@set -e; for f in reports/*binder_sweep*.json; do \
	  if [ -f "$$f" ]; then \
	    echo "Binder: $$f"; \
	    python3 paper/scripts/binder_plot.py --input "$$f" --outdir figs --top-k 10 --metric d_ld; \
	  fi; \
	done || true

.PHONY: sharpener_postprocess
sharpener_postprocess:
	@mkdir -p figs
	@set -e; for f in reports/*layer_entropy_scan*.json; do \
	  if [ -f "$$f" ]; then \
	    echo "Sharpener overlay: $$f"; \
	    python3 paper/scripts/entropy_sharpener_plot.py --input "$$f" --out figs/entropy_overlay; \
	  fi; \
	done || true

figures: postprocess

 

# ============================================================================
# REVIEWER BUNDLE: Package key results for external reviewers
# ============================================================================

.PHONY: stage_reports bundle_review

# Stage outputs into conventional subfolders expected by reviewers
stage_reports:
	@mkdir -p reports/tables reports/appendices
	@# Copy summary tables and percentiles
	@cp -f reports/*_summary_table.csv reports/tables/ 2>/dev/null || true
	@cp -f reports/*_trio_percentiles.json reports/tables/ 2>/dev/null || true
	@# Copy head ranking slices
	@cp -f reports/*_head_ranking.csv reports/tables/ 2>/dev/null || true
	@cp -f reports/*_top*.csv reports/tables/ 2>/dev/null || true
	@cp -f reports/*_bottom*.csv reports/tables/ 2>/dev/null || true
	@# Copy OV reports
	@cp -f reports/ov_report_*.json reports/appendices/ 2>/dev/null || true

bundle_review: stage_reports
	@echo "📦 Building reviewer bundle..."
	@mkdir -p build/
	@TAR=build/results_bundle_$(shell date +%Y%m%d).tar.gz; \
	 tar -czf $$TAR \
		reports/RESULTS_MANIFEST.json \
		reports/tables/*.csv \
		reports/tables/*.json \
		reports/appendices/ov_report_*.json \
		reports/README.md \
		docs/RESULTS_MANIFEST.md \
		docs/REPLICATION.md; \
	 if command -v sha256sum >/dev/null 2>&1; then \
	   sha256sum $$TAR > $$TAR.sha256; \
	 else \
	   shasum -a 256 $$TAR > $$TAR.sha256; \
	 fi; \
	 echo "✅ Bundle created: $$TAR"; \
	 echo "   Checksum: $$TAR.sha256"

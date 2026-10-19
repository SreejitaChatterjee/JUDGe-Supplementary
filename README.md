# Code and cache archive

Code and cached tensors for the paper

> **Reliability of LLM-as-a-Judge: Confidence-Signal Disagreement and Response-Order Sensitivity**
> Sreejita Chatterjee, Indian Institute of Technology Dharwad, India.
> JUDGe 2026: Can We Trust the Judge? (NeurIPS 2026 workshop), camera-ready version.

This archive was first released, anonymized, alongside an earlier version of
the paper titled "Beyond Scalar Confidence: A Reliability Framework for
LLM-as-Judge Verdicts". This README maps the **camera-ready** paper's
tables and headline numbers to the command that produced each one and the
cache that command reads. Section, table and appendix numbers below follow the
camera-ready paper.

Every analysis script reads the shipped `data/*.pt` caches and fixes
`seed=0`, so it is deterministic given those caches and runs no model
inference. The cache-building `collect_*.py` scripts run fresh inference and
are not guaranteed to be bit-for-bit reproducible (see **Seeds** below).

## What changed between the submission and the camera-ready

The camera-ready narrows several claims in response to review:

- **Order averaging** (Table 5) and the **sixth combiner feature**
  (swap-consistency, 0.673 to 0.728 AUROC) are reported as exploratory. The
  order-averaging rule was selected on `id_test` among four rules, and the
  sixth feature was added after inspecting a pool that included `id_test`.
- **Splits** for the judge task are row-level: rows of the same question can
  fall in different splits. Only the combiner-vs-best-signal comparison was
  re-checked with a question-level split (`question_level_resplit.py`).
- **Routing** (the Execute/Verify/HITL rule R_tau) is specified but not
  evaluated on the judge task. The submission's routing-cost table and its
  supporting analyses are not reported in the camera-ready; the scripts are
  kept for reference (see "Not reported in the camera-ready" below).
- **Calibration** (ECE/Brier) comparisons are not reported, because they were
  not assessed on a held-out split.
- **Multiplicity:** Holm correction covers only the combiner-vs-best-signal
  family; significance statements for swap-consistency and order averaging
  are not corrected for multiplicity.

The paper's limitations section (item vi) states these in full.

## Quickstart

```bash
pip install -r requirements.txt
python scripts/judge_characterization.py
```

Reproduces the paper's central comparison (Tables 1 and 2, judge rows: MSP,
best single signal, and combiner AUROC with question-id cluster-bootstrap CIs)
from the cached `data/*.pt` tensors, with no model inference. The script also
prints a SmolLM2-360M row, which the paper treats as non-confirmatory and
excludes from the main table.

## Structure

```
scripts/     Analysis and data-collection scripts
src/         The `deployment_reliability` package: signal extraction (Phi),
             combiner (G_w), router (R_tau), calibration, significance tests
tests/       Unit tests for src/deployment_reliability
notebooks/   Exploratory notebooks from earlier development stages
data/        Cached tensors (.pt files), see the mapping below
requirements.txt, requirements-dev.txt, pyproject.toml
```

## Setup

```bash
pip install -r requirements.txt        # or requirements-dev.txt for tests/notebooks/figures
```

All code runs on CPU; nothing in `src/` or `scripts/` needs CUDA. Scripts load
caches through relative `data/<name>.pt` paths, so run every command from the
repository root.

**Not included:** model weights (Qwen2.5-0.5B/1.5B-Instruct,
SmolLM2-360M-Instruct, GPT-2, Pythia-160m, ResNet-50) and raw datasets
(MT-Bench, WikiText-2, ImageNet-1k, Imagenette). All are public and are
downloaded on first use by the `collect_*.py` scripts, or by
`scripts/download_eval_data.py` for the vision datasets.

**Model and data revisions.** The cached tensors were produced from these
Hugging Face revisions. Pass `revision=` to `from_pretrained` / `load_dataset`
to rebuild a cache against the same weights and data:

| Model or dataset | Revision |
|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` |
| `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | `a10cc1512eabd3dde888204e902eca88bddb4951` |
| `gpt2` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| `EleutherAI/pythia-160m` | `50f5173d932e8e61f858120bcb800b97af589f46` |
| `lmsys/mt_bench_human_judgments` (dataset) | `f7d2896d2cc5d80f8b55c2bbc722613555233c25` |

ResNet-50 uses torchvision's `ResNet50_Weights.IMAGENET1K_V2`.

**Seeds:** `seed=0` fixes every split, bootstrap and reseed (sweeps 0-4 where a
script reseeds explicitly, e.g. `stability_across_splits.py`,
`gbt_seed_variance.py`). Re-running a `collect_judge_verdicts*.py` or
`collect_logits*.py` script performs fresh model inference and should match
the cached values up to floating-point noise, not necessarily bit-for-bit.

**Tests:** `pytest -q` from the repository root (with
`requirements-dev.txt`). At the time of submission this passed 273 of 278
tests, with 5 skipped pending optional large downloads (ImageNet-A/O, the full
ImageNet-1k validation set, and Imagenette).

## Section 3 (Method): the verdict-token restriction of Phi

| Command | Output |
|---|---|
| `python scripts/collect_judge_verdicts_2way.py` | `data/judge_feature_cache_mtbench_2way.pt`, `_1p5b_2way.pt`, `_smollm2_360m_2way.pt` (cached): Phi restricted to the two verdict-token logits `[z_A, z_B]` |
| `python scripts/compare_2way_vs_fullvocab.py` | 2-way vs. full-vocabulary MSP, best-feature and combiner AUROC, and MSP discordance, side by side |

## Section 4.1, Tables 1 and 2: cross-domain checks

| Command | Output |
|---|---|
| `python scripts/paper_diagnostics.py` | §1: Table 1 headline rows (MSP, best single signal, combiner AUROC, difference, DeLong p, winner) for ResNet-50, GPT-2, Pythia-160m, Qwen2.5-0.5B (LLM) and the Qwen judges |
| `python scripts/judge_characterization.py` | §A/§B: Tables 1 and 2, judge rows, with question-id cluster-bootstrap CIs |
| `python scripts/paper_ablation_and_effect_size.py` | Table 2: the five oriented single-signal AUROCs and the combiner, per backbone, plus the combiner-vs-MSP paired-bootstrap CI |
| `python scripts/vision_subsample_at_judge_n.py` | Power check: 1000 draws of the vision `id_test` down to n=642 reach p<0.05 in only 42% of draws |
| `python scripts/best_feature_selected_on_cal.py` | Historical only: an earlier select-on-`threshold_cal` protocol for the best single signal. Tables 1 and 2 now select by a pooled bootstrap-majority vote over `combiner_fit`+`threshold_cal` (n=642); this script backs no number in the paper |

Judge rows read `data/judge_feature_cache_mtbench.pt` and `_1p5b.pt` (built by
`collect_judge_verdicts.py` and `collect_judge_verdicts_1p5b.py`).
`collect_judge_verdicts_smollm2.py` (SmolLM2-1.7B) and
`collect_judge_verdicts_7b.py` (Qwen2.5-7B/3B) were stopped as too slow on the
reference machine and back no number in the paper.

**Split caveat:** the judge and language-model splits are row-level (the
language-model split is over individual next-token predictions), and the
ResNet-50 combiner is fitted on an Imagenette-drawn sample and tested on the
ImageNet validation set without ruling out overlap. See the paper's
limitation (vi).

## Table 3 (Section 4.1): pairwise rank discordance

| Command | Output |
|---|---|
| `python scripts/pairwise_rank_discordance.py` | "POOLED-BOOTSTRAP-STABILIZED orientation" block: Table 3 |
| `python scripts/direction_split_robustness_check.py` | Exploratory check of the per-configuration direction convention used in Table 3; backs no number in the paper |

## Section 4.2: selective judging as a proposed, unevaluated procedure

The camera-ready specifies the routing rule R_tau and derives its thresholds
(Appendix A.3) but does not evaluate it on the judge task. No script result is
reported for this section. The submission's routing analyses are listed under
"Not reported in the camera-ready" below.

## Section 5.1, Tables 4 and 5

| Command | Output |
|---|---|
| `python scripts/equivalence_test_tost.py` | TOST equivalence test (row-level split), both margins (0.02 AUROC and the configuration's own MDE), both judge sizes |
| `python scripts/failure_mode_table.py` | Table 4: all six cells (high-confidence-wrong, low-confidence-correct, high-disagreement, at 0.5B and 1.5B) |
| `python scripts/collect_judge_swap_consistency.py`, `scripts/collect_judge_swap_consistency_1p5b.py` | `data/judge_swap_consistency_cache.pt`, `_1p5b.pt` (cached): the explicit order-swap test at both judge sizes |
| `python scripts/order_averaging_correction.py` | Binary-verdict order averaging (uninformative: verdicts flip on 98.9% of pairs at 0.5B); also the coin-flip fallback accuracy, 49.9% |
| `python scripts/collect_judge_swap_consistency_continuous.py` | `data/judge_swap_consistency_cache_continuous.pt` (cached): the 0.5B swap test with `phi_swapped` persisted for continuous averaging |
| `python scripts/order_averaging_continuous_analysis.py` | Table 5 (exploratory; rule selected on `id_test` among four rules): single-order 50.9% to two-pass order-averaged 64.0% (cluster-bootstrap 95% CI [59.0%, 68.7%]; McNemar p=2.9e-6, not corrected for the selection) |
| `python scripts/order_averaging_auroc_check.py` | AUROC side of Table 5 (0.5B, `id_test`, n=642): 0.552 to 0.645; paired cluster-bootstrap CI on the gap [0.031, 0.157] |
| `python scripts/verbosity_and_swap_crosstab.py` | Part A: the verbosity check (predicted winner matches the longer response at about 51.9%/51.0%, vs. 70.8% for the human label); Part B: swap-consistency crossed with the Table 4 failure modes |

## Appendix A.1, Table 6: robustness and scope

Table 6 summarizes results produced by the scripts in the other sections
(question-level resplit, 2-way Phi, failure modes, swap tests, sixth-feature
check, 5-reseed stability). It has no separate script.

## Appendix A.3: router threshold derivation

Analytical derivation in the paper; no script.

## Appendix A.4: extended backbone results (GPT-2, Pythia-160m)

| Command | Output |
|---|---|
| `python scripts/paper_diagnostics.py` | §3/§10: standardized-refit AUROC (0.829 to 0.846) and nonlinear-combiner AUROC per LLM backbone |
| `python scripts/combiner_regularization_sweep.py` | L2-penalty sweep: whether regularization strength alone explains GPT-2's gap vs. Pythia-160m needing standardization |

## Appendix A.5: extended judge characterization

| Command | Output |
|---|---|
| `python scripts/inter_annotator_agreement.py` | 274/351 = 78.1% item-level and 633/772 = 82.0% pairwise annotator agreement |
| `python scripts/majority_label_robustness_check.py` | Majority-vote relabeling of the multiply-annotated subset (n=391 of 642 `id_test` rows): combiner-vs-best-signal gap +0.030 vs. +0.031 at 0.5B and -0.010 vs. -0.005 at 1.5B |
| `python scripts/stability_across_splits.py` | 5-reseed combiner-vs-MSP stability, LLM and judge backbones |
| `python scripts/vision_seed_stability.py` | The same 5-reseed treatment for the vision row |
| `python scripts/power_analysis.py` | Minimum detectable effect at 80% power: 0.036 (0.5B) and 0.018 (1.5B), computed with the naive DeLong SE |
| `python scripts/nonlinear_combiner_judge_task.py` | Untuned GBT combiner vs. the linear combiner on the judge task (the paper reports AUROC only) |
| `python scripts/gbt_seed_variance.py` | 5-seed variance for the GBT combiner on the three LLM backbones |
| `python scripts/matched_subset_reslice.py` | Re-slices the Qwen cache to the SmolLM2 subset for a like-for-like comparison |
| `python scripts/question_level_resplit.py` | Question-level split (32/8/40 of 80 questions; 546/121/617 rows), combiner-vs-best-signal only: +0.050 (p=0.051) at 0.5B, -0.007 (p=0.169) at 1.5B |
| `python scripts/selection_aware_bootstrap.py` | Best-signal selection noise (limitation iv), holding `id_test` fixed and resampling the calibration pool |
| `python scripts/sixth_signal_cluster_check.py` | Exploratory 5-vs-6-feature combiner gap at 1.5B: 0.673 to 0.728, question-id cluster-bootstrap CI [0.030, 0.081] with the fitted combiners held fixed (row-level split; not independently confirmed) |
| `python scripts/swap_consistency_auroc_ci_1p5b.py` | Standalone swap-consistency AUROC at 1.5B on the full pool: 0.583, cluster-bootstrap CI [0.551, 0.613] |
| `python scripts/swap_consistency_single_auroc_significance_0p5b.py` | Single-sample DeLong test of the standalone swap-consistency AUROC at 0.5B (0.503): z=0.97, p=0.33 |

## Not reported in the camera-ready

These scripts back results from the submitted version that the camera-ready
does not report. They are kept for reference only.

| Command | Submitted-version result | Why not reported |
|---|---|---|
| `python scripts/selective_judging.py`, `scripts/selective_judging_calibrated_msp.py` | Routing-cost table (MSP, Platt-scaled MSP, combiner, and always-act baselines) | The routed combiner score was not recalibrated, costs were assumed, the second forward pass was not counted, and no Verify/HITL outcome was observed |
| `python scripts/routing_multiplicity_check.py` | Bonferroni family for the routing comparisons | Depends on the routing table above |
| `python scripts/selective_judging_utility_and_sweep.py` | Reward-term robustness and the 20-cell cost-regime sweep | Depends on the routing table above |
| `python scripts/calibration_reliability_bins.py` | ECE/Brier summary and reliability bins | Calibration was not assessed on a held-out split |

## Reproducing a headline number (example)

```python
import torch
d = torch.load("data/judge_feature_cache_mtbench_1p5b.pt", weights_only=False)
# d contains Phi feature tensors, correctness labels, and the fit/cal/test split assignment
```

`scripts/judge_characterization.py` and `scripts/paper_diagnostics.py` hold the
AUROC, DeLong and cluster-bootstrap code used throughout; the other analysis
scripts import one or the other.

## Which cache backs which result

| Cache file | Backs |
|---|---|
| `judge_feature_cache_mtbench.pt` | Qwen2.5-0.5B-Instruct judge: Phi and correctness (Tables 1-4) |
| `judge_feature_cache_mtbench_1p5b.pt` | Qwen2.5-1.5B-Instruct judge: same |
| `judge_feature_cache_mtbench_2way.pt`, `_1p5b_2way.pt` | 2-way Phi restriction (Section 3, Appendix A.5) |
| `judge_feature_cache_mtbench_smollm2_360m.pt` (+`_2way`) | SmolLM2-360M non-confirmatory check (Section 4.1) |
| `judge_swap_consistency_cache.pt`, `_1p5b.pt` | Order-swap test (Section 5.1) |
| `judge_swap_consistency_cache_continuous.pt` | Continuous order averaging (Table 5) |
| `logit_cache_resnet50.pt` | Vision backbone fit/calibration sample (Imagenette) |
| `logit_cache_imagenet1k_resnet50.pt` | 50,000-image ImageNet-1k `id_test` (Table 1, ResNet-50 row) |
| `llm_feature_cache_gpt2.pt`, `_pythia160m.pt`, `_qwen05binstruct.pt` | Language-model backbones on WikiText-2 (Table 1, Appendix A.4) |

Other files in `data/` (`logit_cache_vit_b16.pt`, `_convnext_tiny.pt`, the
DTD/iNaturalist/Places365/ImageNet-C variants, `mahalanobis_feature_cache_*`,
`odin_cache_*`, `cross_model_correctness_results*`, `llm_free_running_cache_*`)
come from earlier development of the `deployment_reliability` pipeline and back
no number in the paper.

## Using the signals on your own judge

The package installs directly from this repository:

```bash
pip install git+https://github.com/research-codebase-095/JUDGe-Supplementary
```

```python
from deployment_reliability import featurize, DEFAULT_FEATURE_NAMES

# logits: the judge's full next-token logits at the verdict position, shape (n, vocab)
phi = featurize(logits, normalize_l2=True)   # (n, 5): msp, logit_margin, normalized_entropy, energy_score, logit_l2_norm
verdict_a = logits[:, token_id_a] > logits[:, token_id_b]

# swap-consistency: rerun with the two responses swapped, map the verdict back
# to the original labelling, and compare
swap_consistent = verdict_a == ~verdict_a_swapped
```

`scripts/collect_judge_verdicts.py` shows the full prompt, chat template and
verdict-token extraction used in the paper.

## Citation

GitHub's "Cite this repository" button reads `CITATION.cff`.

```bibtex
@inproceedings{chatterjee2026reliability,
  title     = {Reliability of {LLM}-as-a-Judge: Confidence-Signal Disagreement and Response-Order Sensitivity},
  author    = {Chatterjee, Sreejita},
  booktitle = {JUDGe 2026: Can We Trust the Judge? (NeurIPS 2026 Workshop)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=9lIJ2lRhwC}
}
```

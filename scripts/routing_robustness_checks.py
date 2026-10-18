"""Routing and combiner robustness checks reported in the appendix (print-only).

All checks reuse the existing feature caches; no model inference.

1. Cost sweep: for each cell of c_E x c_H, both the thresholds AND the realized
   costs follow that cell's cost table, and each routed policy (combiner,
   Platt-scaled MSP) is compared with the cheapest fixed policy in the cell
   (always-verify, always-HITL or always-execute) by a paired question-id
   cluster bootstrap (1,000 resamples). This supersedes task B of
   selective_judging_utility_and_sweep.py, which realizes costs under the base
   table in every cell.
2. Reward term: a correct Execute costs -u (u=1), credited in the realized cost.
3. Standardized combiner vs best single signal (paired cluster bootstrap).
4. Verdict inversion: acting on the more likely verdict, and how much of the
   router's advantage over always-verify comes from HITL vs Execute rows.
5. Sensitivity to the Platt fit: refit on 1,000 question-cluster resamples of
   threshold_cal, route id_test, compare with always-verify.
6. Calibration (ECE with 10 equal-width bins, Brier) on id_test.
7. Routing under the question-level split of question_level_resplit.py,
   including the 4-comparison Bonferroni (98.75%) interval.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from deployment_reliability.calibration import PlattScaling  # noqa: E402
from deployment_reliability.combiner import LogisticRegressionCombiner  # noqa: E402
from deployment_reliability.router import cost_sensitive_thresholds  # noqa: E402
from judge_characterization import (  # noqa: E402
    best_single_feature, cluster_bootstrap_auroc_diff, load_all_judge_configs, standardized_combiner,
)
from question_level_resplit import question_level_split  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
CONFIGS = [("judge_feature_cache_mtbench.pt", "0.5B"), ("judge_feature_cache_mtbench_1p5b.pt", "1.5B")]
BASE = dict(c_e=10.0, c_v=1.0, c_vp=3.0, c_h=2.0)
C_E_GRID = (5, 7.5, 10, 15, 20)
C_H_GRID = (1, 1.5, 2, 3, 4)


def load(fname: str):
    cache = torch.load(os.path.join(DATA_DIR, fname))
    return cache["phi"], cache["correct"].bool(), np.array(cache["question_id"]), np.array(cache["splits"])


def thresholds(c_e, c_v, c_vp, c_h, u=0.0):
    return cost_sensitive_thresholds(c_execute_incorrect=c_e, c_verify_correct=c_v,
                                     c_verify_incorrect=c_vp, c_hitl=c_h, u_execute_correct=u)


def route_costs(p, correct, c_e, c_v, c_vp, c_h, u=0.0):
    """Realized per-row cost of the two-threshold router under THIS cost table."""
    hi, lo = thresholds(c_e, c_v, c_vp, c_h, u)
    ex, hitl = p >= hi, p < lo
    ver = ~ex & ~hitl
    cost = np.where(ex, np.where(correct, -u, c_e), 0.0)
    cost = np.where(ver, np.where(correct, c_v, c_vp), cost)
    cost = np.where(hitl, c_h, cost)
    return cost, int(ex.sum()), int(ver.sum()), int(hitl.sum())


def verify_costs(correct, c_v=1.0, c_vp=3.0):
    return np.where(correct, c_v, c_vp)


def paired_ci(qid, a, b, n=1000, alpha=0.05, seed=0):
    """Question-id cluster bootstrap of mean(a - b)."""
    uq = np.unique(qid)
    idx = {q: np.where(qid == q)[0] for q in uq}
    rng = np.random.default_rng(seed)
    d = a - b
    draws = np.empty(n)
    for i in range(n):
        pick = np.concatenate([idx[q] for q in rng.choice(uq, size=len(uq), replace=True)])
        draws[i] = d[pick].mean()
    return d.mean(), np.quantile(draws, alpha / 2), np.quantile(draws, 1 - alpha / 2)


def ece(p, y, bins=10):
    total = 0.0
    for k in range(bins):
        m = (p >= k / bins) & ((p < (k + 1) / bins) if k < bins - 1 else (p <= 1))
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return total


def row_level(fname: str) -> dict:
    phi, correct, qid, splits = load(fname)
    mf, mc, mt = splits == "combiner_fit", splits == "threshold_cal", splits == "id_test"
    comb = LogisticRegressionCombiner().fit(phi[mf], correct[mf].float())
    platt = PlattScaling().fit(phi[mc][:, 0], correct[mc].float())
    return dict(phi=phi, correct=correct, qid=qid, mc=mc, mt=mt,
                comb=comb.score(phi[mt]).numpy(), platt=platt.transform(phi[mt][:, 0]).numpy(),
                msp=phi[mt][:, 0].numpy(), y=correct[mt].numpy(), q=qid[mt])


def check_cost_sweep(D):
    print("=" * 90, "\n1. COST SWEEP vs cheapest fixed policy per cell (costs realized under each cell's table)")
    for name, d in D.items():
        y, q = d["y"], d["q"]
        for pol in ("comb", "platt"):
            win, ns, lose = [], [], []
            for c_e in C_E_GRID:
                for c_h in C_H_GRID:
                    if not (0 <= 1 <= c_h <= 3 < c_e):
                        continue
                    base = {"verify": verify_costs(y), "hitl": np.full(len(y), float(c_h)),
                            "execute": np.where(y, 0.0, float(c_e))}
                    bname = min(base, key=lambda k: base[k].mean())
                    rc, *_ = route_costs(d[pol], y, c_e, 1.0, 3.0, c_h)
                    pt, lo, hi = paired_ci(q, rc, base[bname])
                    (win if hi < 0 else lose if lo > 0 else ns).append((c_e, c_h, bname))
            print(f"{name} {pol:<5}: cheaper than best fixed policy in {len(win)}/20 cells {win}; "
                  f"n.s. {len(ns)}; more expensive {len(lose)}")


def check_reward_term(D):
    print("=" * 90, "\n2. REWARD TERM (correct Execute costs -u)")
    for name, d in D.items():
        for pol in ("comb", "platt"):
            for u in (0.0, 1.0):
                hi, lo = thresholds(**BASE, u=u)
                rc, ne, nv, nh = route_costs(d[pol], d["y"], **BASE, u=u)
                pt, l95, h95 = paired_ci(d["q"], rc, verify_costs(d["y"]))
                print(f"{name} {pol:<5} u={u}: tau_hi={hi:.3f} tau_lo={lo:.3f} exec={ne} verify={nv} hitl={nh} "
                      f"minus always-verify {pt:+.4f} [{l95:+.4f},{h95:+.4f}]")


def check_standardized_combiner():
    print("=" * 90, "\n3. STANDARDIZED COMBINER vs BEST SINGLE SIGNAL")
    for cfg in load_all_judge_configs()[:2]:
        std_c, _, phi_test_std = standardized_combiner(cfg)
        bname, bscores = best_single_feature(cfg)
        pt, lo, hi, frac = cluster_bootstrap_auroc_diff(cfg["question_id"], cfg["correct"].numpy(),
                                                       std_c.score(phi_test_std).numpy(), bscores.numpy())
        print(f"{cfg['name']}: standardized combiner minus {bname}: {pt:+.4f} [{lo:+.4f},{hi:+.4f}]")


def check_inversion(D):
    print("=" * 90, "\n4. VERDICT INVERSION and source of the routing advantage (base cost table)")
    hi, lo = thresholds(**BASE)
    for name, d in D.items():
        vc = verify_costs(d["y"])
        for pol in ("comb", "platt"):
            p, y = d[pol], d["y"]
            rc, *_ = route_costs(p, y, **BASE)
            flip = p < 0.5
            ic, ne, nv, nh = route_costs(np.where(flip, 1 - p, p), np.where(flip, ~y, y), **BASE)
            hitl_m, ex_m = p < lo, p >= hi
            pt, l95, h95 = paired_ci(d["q"], ic, rc)
            print(f"{name} {pol:<5}: router {rc.mean():.3f}, inversion {ic.mean():.3f} "
                  f"(inversion minus router {pt:+.4f} [{l95:+.4f},{h95:+.4f}]) | router minus always-verify from "
                  f"HITL rows {(rc - vc)[hitl_m].sum() / len(y):+.4f}, Execute rows {(rc - vc)[ex_m].sum() / len(y):+.4f} | "
                  f"accuracy of HITL-routed verdicts {y[hitl_m].mean():.3f} (n={int(hitl_m.sum())})")


def check_platt_sensitivity(D, n_refit=1000):
    print("=" * 90, f"\n5. SENSITIVITY TO THE PLATT FIT ({n_refit} question-cluster resamples of threshold_cal)")
    for name, d in D.items():
        phi, correct, qid, mc, mt = d["phi"], d["correct"], d["qid"], d["mc"], d["mt"]
        cal_rows = np.where(mc)[0]
        uq = np.unique(qid[cal_rows])
        rng = np.random.default_rng(0)
        vc = verify_costs(d["y"])
        diffs, n_hitl = [], []
        for _ in range(n_refit):
            rows = np.concatenate([cal_rows[qid[cal_rows] == q] for q in rng.choice(uq, size=len(uq), replace=True)])
            platt = PlattScaling().fit(phi[rows][:, 0], correct[rows].float())
            rc, _, _, nh = route_costs(platt.transform(phi[mt][:, 0]).numpy(), d["y"], **BASE)
            diffs.append((rc - vc).mean())
            n_hitl.append(nh)
        diffs, n_hitl = np.array(diffs), np.array(n_hitl)
        print(f"{name}: Platt-routed minus always-verify: median {np.median(diffs):+.4f}, "
              f"2.5-97.5% [{np.quantile(diffs, .025):+.4f},{np.quantile(diffs, .975):+.4f}], "
              f"share < 0: {np.mean(diffs < 0):.3f} | HITL count {int(np.quantile(n_hitl, .025))}-{int(np.quantile(n_hitl, .975))}")


def check_calibration(D):
    print("=" * 90, "\n6. CALIBRATION ON id_test")
    for name, d in D.items():
        y = d["y"].astype(float)
        for pol in ("msp", "platt", "comb"):
            p = d[pol]
            print(f"{name} {pol:<5}: ECE={ece(p, y):.4f} Brier={np.mean((p - y) ** 2):.4f}")


def check_question_level_routing():
    print("=" * 90, "\n7. ROUTING UNDER THE QUESTION-LEVEL SPLIT")
    for fname, name in CONFIGS:
        phi, correct, qid, _ = load(fname)
        mf, mc, mt, nf, nc, nt, _ = question_level_split(qid)
        comb = LogisticRegressionCombiner().fit(phi[mf], correct[mf].float())
        platt = PlattScaling().fit(phi[mc][:, 0], correct[mc].float())
        y, q = correct[mt].numpy(), qid[mt]
        vc = verify_costs(y)
        print(f"{name}: questions {nf}/{nc}/{nt}, rows {mf.sum()}/{mc.sum()}/{mt.sum()}, always-verify {vc.mean():.3f}")
        for pol, p in (("comb", comb.score(phi[mt]).numpy()), ("platt", platt.transform(phi[mt][:, 0]).numpy())):
            rc, ne, nv, nh = route_costs(p, y, **BASE)
            pt, lo, hi = paired_ci(q, rc, vc)
            _, lo_b, hi_b = paired_ci(q, rc, vc, n=5000, alpha=0.05 / 4)
            print(f"   {pol:<5} cost {rc.mean():.3f} | minus always-verify {pt:+.4f} 95% [{lo:+.4f},{hi:+.4f}] "
                  f"98.75% [{lo_b:+.4f},{hi_b:+.4f}]")


def main() -> None:
    D = {name: row_level(fname) for fname, name in CONFIGS}
    check_cost_sweep(D)
    check_reward_term(D)
    check_standardized_combiner()
    check_inversion(D)
    check_platt_sensitivity(D)
    check_calibration(D)
    check_question_level_routing()


if __name__ == "__main__":
    main()

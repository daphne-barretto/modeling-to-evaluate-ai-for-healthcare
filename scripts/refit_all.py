"""One-shot refit driver: imports heavy deps once, then runs every analysis.

Pays the cold-start torch/pandas import cost a single time instead of once
per `python -m analysis.X` invocation (which on this machine is ~45 minutes
when the disk cache is cold). Each step is gated by a try/except so a single
late-stage failure doesn't lose the earlier work.

Order is: data prep -> IRT fits -> downstream analyses that depend on them
-> figures -> manuscript-numbers export.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def step(label: str, fn):
    print(f"\n{'='*72}\n=  {label}\n{'='*72}", flush=True)
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        print(f"\n[OK] {label}  ({dt:.1f}s)", flush=True)
        return True
    except Exception:
        dt = time.time() - t0
        print(f"\n[FAIL] {label}  ({dt:.1f}s)", flush=True)
        traceback.print_exc()
        return False


def main():
    t_total = time.time()

    from analysis import fit_irt
    step("fit_irt", fit_irt.main)

    from analysis import fit_irt_by_tier
    step("fit_irt_by_tier", fit_irt_by_tier.main)

    from analysis import bifactor_mirt
    step("bifactor_mirt", bifactor_mirt.main)

    from analysis import baselines
    step("baselines", baselines.main)

    from analysis import reliability
    step("reliability", reliability.main)

    from analysis import dif_analysis
    step("dif_analysis", dif_analysis.main)

    from analysis import scaling_law
    step("scaling_law", scaling_law.main)

    from analysis import tinybenchmark
    step("tinybenchmark", tinybenchmark.main)

    from analysis import irt_vs_baseline
    step("irt_vs_baseline", irt_vs_baseline.main)

    from analysis import amortized_irt
    step("amortized_irt", amortized_irt.main)

    from analysis import figures
    step("figures", figures.main)

    from analysis import figures_by_tier
    step("figures_by_tier", figures_by_tier.main)

    from analysis import export_manuscript_numbers
    step("export_manuscript_numbers", export_manuscript_numbers.main)

    print(f"\nTOTAL: {time.time() - t_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()

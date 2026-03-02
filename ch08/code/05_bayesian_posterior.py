import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def probabilities_from_draws(draws: Iterable[float], thresholds: list[float]) -> dict[str, float]:
    draws_list = list(draws)
    if not draws_list:
        raise ValueError("No draws provided.")

    n = len(draws_list)
    probs: dict[str, float] = {}
    probs["P(tau<0)"] = sum(d < 0 for d in draws_list) / n
    for thr in thresholds:
        probs[f"P(tau<{thr})"] = sum(d < thr for d in draws_list) / n
    return probs


def posterior_normal_from_prior(
    mu_hat: float,
    se_hat: float,
    prior_mean: float,
    prior_sd: float,
) -> tuple[float, float]:
    if se_hat <= 0:
        raise ValueError("se_hat must be > 0")
    if prior_sd <= 0:
        raise ValueError("prior_sd must be > 0")

    prior_var = prior_sd**2
    like_var = se_hat**2

    post_var = 1.0 / (1.0 / prior_var + 1.0 / like_var)
    post_mean = post_var * (prior_mean / prior_var + mu_hat / like_var)
    return post_mean, math.sqrt(post_var)


def probabilities_from_normal(mu: float, sd: float, thresholds: list[float]) -> dict[str, float]:
    if sd <= 0:
        raise ValueError("sd must be > 0")

    probs: dict[str, float] = {}
    probs["P(tau<0)"] = normal_cdf((0.0 - mu) / sd)
    for thr in thresholds:
        probs[f"P(tau<{thr})"] = normal_cdf((thr - mu) / sd)
    return probs


def load_sdid_module(module_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("sdid_analysis", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SDID 결과를 베이지언(사후확률) 형태로 요약하고 bootstrap τ 분포를 저장합니다."
    )
    parser.add_argument("--zeta", type=float, default=0.01, help="SDID 단위가중치 L2 페널티 (02_sdid_analysis.py)")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[-0.5, -1.0],
        help="P(tau<threshold)로 보고할 임계값들",
    )
    parser.add_argument(
        "--prior-sds",
        type=float,
        nargs="*",
        default=[0.5, 1.0, 2.0],
        help="정규 사전분포 N(0, sd^2) 후보들",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    sdid_module = load_sdid_module(base_dir / "02_sdid_analysis.py")

    df_wide = pd.read_csv(base_dir / "panel_data_wide.csv", index_col=0)

    sdid_results = sdid_module.synthetic_did(df_wide, zeta=args.zeta)
    mu_hat = float(sdid_results["effect"])
    se_hat = float(sdid_results["se"])
    ci_lower = float(sdid_results["ci_lower"])
    ci_upper = float(sdid_results["ci_upper"])

    bootstrap_effects = list(map(float, sdid_results.get("bootstrap_effects", [])))
    if not bootstrap_effects:
        raise RuntimeError(
            "bootstrap_effects가 비어 있습니다. sdid/02_sdid_analysis.py의 synthetic_did 결과에 "
            "'bootstrap_effects'가 포함되는지 확인하세요."
        )

    bootstrap_df = pd.DataFrame({"draw": range(1, len(bootstrap_effects) + 1), "tau": bootstrap_effects})
    bootstrap_csv = base_dir / "bootstrap_effects.csv"
    bootstrap_df.to_csv(bootstrap_csv, index=False, encoding="utf-8-sig")

    summary: dict[str, object] = {
        "point_estimate": {"tau_hat": mu_hat, "se_bootstrap": se_hat, "ci95_bootstrap": [ci_lower, ci_upper]},
        "draws": {"n_bootstrap": len(bootstrap_effects), "file": str(bootstrap_csv.name)},
        "probabilities": {},
    }

    summary["probabilities"]["empirical_from_bootstrap"] = probabilities_from_draws(
        bootstrap_effects, thresholds=list(args.thresholds)
    )
    summary["probabilities"]["normal_approx_flat_prior"] = probabilities_from_normal(
        mu_hat, se_hat, thresholds=list(args.thresholds)
    )

    prior_results = []
    for prior_sd in args.prior_sds:
        post_mean, post_sd = posterior_normal_from_prior(
            mu_hat=mu_hat, se_hat=se_hat, prior_mean=0.0, prior_sd=float(prior_sd)
        )
        prior_results.append(
            {
                "prior": {"mean": 0.0, "sd": float(prior_sd)},
                "posterior": {"mean": post_mean, "sd": post_sd},
                "probabilities": probabilities_from_normal(post_mean, post_sd, thresholds=list(args.thresholds)),
            }
        )
    summary["probabilities"]["normal_with_priors"] = prior_results

    out_json = base_dir / "bayesian_posterior_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = base_dir / "bayesian_posterior_summary.csv"
    rows = [
        {"model": "empirical_from_bootstrap", **summary["probabilities"]["empirical_from_bootstrap"]},
        {"model": "normal_approx_flat_prior", **summary["probabilities"]["normal_approx_flat_prior"]},
    ]
    for item in prior_results:
        label = f"normal_prior_sd={item['prior']['sd']}"
        rows.append({"model": label, **item["probabilities"]})
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.hist(bootstrap_effects, bins=30, alpha=0.85, color="#4C78A8", edgecolor="white")
        plt.axvline(0, color="black", linewidth=1, linestyle="--", label="tau=0")
        plt.axvline(mu_hat, color="#F58518", linewidth=2, label=f"tau_hat={mu_hat:.3f}")
        for thr in args.thresholds:
            plt.axvline(thr, color="#E45756", linewidth=1, linestyle=":", label=f"threshold={thr}")
        plt.title("SDID bootstrap τ 분포")
        plt.xlabel("τ (처치효과)")
        plt.ylabel("빈도")
        plt.legend()
        plt.tight_layout()
        plt.savefig(base_dir / "posterior_tau_hist.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    print(f"[OK] bootstrap τ 저장: {bootstrap_csv}")
    print(f"[OK] 요약(JSON): {out_json}")
    print(f"[OK] 요약(CSV): {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


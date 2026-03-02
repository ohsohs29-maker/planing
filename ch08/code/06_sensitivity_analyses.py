import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_sdid_module(module_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("sdid_analysis", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probabilities_from_draws(draws: list[float], thresholds: list[float]) -> dict[str, float]:
    n = len(draws)
    probs: dict[str, float] = {"P(tau<0)": float(np.mean(np.array(draws) < 0))}
    for thr in thresholds:
        probs[f"P(tau<{thr})"] = float(np.mean(np.array(draws) < thr))
    probs["n_draws"] = n
    return probs


def run_sdid(module, df_wide: pd.DataFrame, zeta: float) -> dict:
    results = module.synthetic_did(df_wide, zeta=zeta)
    return results


def apply_transform(df_wide: pd.DataFrame, transform: str) -> pd.DataFrame:
    if transform == "level":
        return df_wide
    if transform == "log":
        values = df_wide.astype(float).values
        if np.any(values <= 0):
            raise ValueError("log 변환은 값이 0 이하이면 적용할 수 없습니다.")
        return pd.DataFrame(np.log(values), index=df_wide.index, columns=df_wide.columns)
    raise ValueError(f"Unknown transform: {transform}")


def main() -> int:
    parser = argparse.ArgumentParser(description="SDID 민감도 분석(기간 절단/비교군 제외/변환)을 자동 실행합니다.")
    parser.add_argument("--zeta", type=float, default=0.01, help="SDID 단위가중치 L2 페널티")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[-0.5, -1.0],
        help="P(tau<threshold)로 보고할 임계값들",
    )
    parser.add_argument(
        "--exclude-donors",
        type=str,
        nargs="*",
        default=["경기대학교"],
        help="오염/처치 가능성 때문에 제외할 donor 대학명들(기본: 경기대학교)",
    )
    parser.add_argument(
        "--cutoff-year",
        type=int,
        default=2024,
        help="민감도에서 사용할 연도 절단(기본: 2024까지 사용하여 2025 제외 시나리오 생성)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    module = load_sdid_module(base_dir / "02_sdid_analysis.py")
    df_wide_full = pd.read_csv(base_dir / "panel_data_wide.csv", index_col=0)

    scenarios = [
        {"name": "baseline_level_all_years", "cutoff_year": None, "exclude_donors": [], "transform": "level"},
        {
            "name": f"cutoff_{args.cutoff_year}_level",
            "cutoff_year": args.cutoff_year,
            "exclude_donors": [],
            "transform": "level",
        },
        {
            "name": "exclude_donors_level",
            "cutoff_year": None,
            "exclude_donors": list(args.exclude_donors),
            "transform": "level",
        },
        {
            "name": f"cutoff_{args.cutoff_year}_exclude_donors_level",
            "cutoff_year": args.cutoff_year,
            "exclude_donors": list(args.exclude_donors),
            "transform": "level",
        },
        {"name": "baseline_log_all_years", "cutoff_year": None, "exclude_donors": [], "transform": "log"},
    ]

    outputs = []
    for scenario in scenarios:
        df = df_wide_full.copy()

        if scenario["cutoff_year"] is not None:
            years = [int(c) for c in df.columns]
            keep_cols = [str(y) for y in years if y <= int(scenario["cutoff_year"])]
            df = df[keep_cols]

        if scenario["exclude_donors"]:
            for donor in scenario["exclude_donors"]:
                if donor in df.index and donor != "한신대학교":
                    df = df.drop(index=donor)

        df = apply_transform(df, scenario["transform"])

        res = run_sdid(module, df, zeta=float(args.zeta))
        draws = list(map(float, res.get("bootstrap_effects", [])))
        probs = probabilities_from_draws(draws, thresholds=list(args.thresholds)) if draws else {}

        outputs.append(
            {
                "scenario": scenario["name"],
                "n_units": int(df.shape[0]),
                "n_years": int(df.shape[1]),
                "transform": scenario["transform"],
                "excluded_donors": ",".join(scenario["exclude_donors"]),
                "tau_hat": float(res["effect"]),
                "se_bootstrap": float(res["se"]),
                "ci95_lower": float(res["ci_lower"]),
                "ci95_upper": float(res["ci_upper"]),
                **probs,
            }
        )

    summary_df = pd.DataFrame(outputs)
    out_csv = base_dir / "sensitivity_summary.csv"
    summary_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_json = base_dir / "sensitivity_summary.json"
    out_json.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        x = np.arange(len(summary_df))
        y = summary_df["tau_hat"].values
        yerr_lower = y - summary_df["ci95_lower"].values
        yerr_upper = summary_df["ci95_upper"].values - y
        plt.errorbar(x, y, yerr=[yerr_lower, yerr_upper], fmt="o", capsize=4, color="#4C78A8")
        plt.axhline(0, color="black", linewidth=1, linestyle="--")
        plt.xticks(x, summary_df["scenario"].tolist(), rotation=30, ha="right")
        plt.ylabel("τ (처치효과)")
        plt.title("SDID 민감도 분석: 추정치와 95% CI(bootstrap)")
        plt.tight_layout()
        plt.savefig(base_dir / "sensitivity_effects.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    print(f"[OK] 민감도 요약(CSV): {out_csv}")
    print(f"[OK] 민감도 요약(JSON): {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


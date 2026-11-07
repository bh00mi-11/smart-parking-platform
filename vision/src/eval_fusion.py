"""
Fusion calibration evaluation harness on synthetic trajectory streams.
NOTE: This validates the fusion estimation method and uncertainty propagation on synthetic data,
NOT real-world facility accuracy.

Evaluates on HELD-OUT seeds (tuning was conducted strictly on tuning seeds):
1. Occupancy MAE:
   - Fusion estimate vs True Occupancy
   - Baseline 1 (Last observation only) vs True Occupancy
   - Baseline 2 (Historical prior only) vs True Occupancy
2. Empirical Coverage of the 80% range:
   - Overall coverage percentage
   - Coverage broken down by data-age buckets: [<5 min, 5-15 min, 15-30 min, >30 min]
3. Plots:
   - Reliability / calibration curve
   - 80% interval width vs data age
"""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np

_src_dir = Path(__file__).resolve().parent
_vision_dir = _src_dir.parent
if str(_vision_dir) not in sys.path:
    sys.path.insert(0, str(_vision_dir))

from src.fusion import FusionConfig, estimate_occupancy
from src.sim_stub import simulate_zone_trajectory

# SEED PARTITIONING:
# Tuning seeds: [101, 102]
# HELD-OUT evaluation seeds: [201, 202, 203]
HELD_OUT_SEEDS = [201, 202, 203]


def run_fusion_evaluation(
    output_dir: Union[str, Path] = "vision/outputs",
    config: Optional[FusionConfig] = None
) -> Dict:
    if config is None:
        config = FusionConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("FUSION CALIBRATION & ACCURACY EVALUATION (HELD-OUT SEEDS)")
    print("Notice: Validates fusion method on synthetic simulation; not real-world accuracy.")
    print(f"Seeds: {HELD_OUT_SEEDS} | Zone Types: ['managed', 'open']")
    print("=" * 65)

    zones = [
        {"facility_id": "FAC-TEST", "zone_id": "ZONE-M1", "capacity": 60, "zone_type": "managed"},
        {"facility_id": "FAC-TEST", "zone_id": "ZONE-O1", "capacity": 80, "zone_type": "open"}
    ]

    all_fusion_errors = []
    all_last_obs_errors = []
    all_prior_errors = []

    # Coverage statistics
    covered_count = 0
    total_eval_points = 0

    age_buckets = {
        "<5min": {"total": 0, "covered": 0, "widths": []},
        "5-15min": {"total": 0, "covered": 0, "widths": []},
        "15-30min": {"total": 0, "covered": 0, "widths": []},
        ">30min": {"total": 0, "covered": 0, "widths": []}
    }

    age_to_widths = []

    for seed in HELD_OUT_SEEDS:
        for z in zones:
            history = simulate_zone_trajectory(
                zone_id=z["zone_id"],
                facility_id=z["facility_id"],
                capacity=z["capacity"],
                zone_type=z["zone_type"],
                duration_days=2,
                step_minutes=10,
                camera_interval_minutes=35,
                manual_interval_hours=6,
                camera_noise_std=1.2,
                manual_noise_std=0.8,
                event_drop_rate=0.10,
                seed=seed
            )

            for step in history:
                now_t = step["timestamp"]
                true_occ = step["true_occupancy"]
                obs = step["observations"]
                true_free = z["capacity"] - true_occ

                # 1. Fusion prediction
                update = estimate_occupancy(z, obs, now=now_t, config=config)
                est_occ = z["capacity"] - ((update.free_low + update.free_high) / 2.0)
                fusion_err = abs(est_occ - true_occ)
                all_fusion_errors.append(fusion_err)

                # 2. Baseline 1: Last observation only
                last_obs_occ = None
                if "camera" in obs and obs["camera"] is not None:
                    last_obs_occ = obs["camera"].get("count")
                elif "manual" in obs and obs["manual"] is not None:
                    last_obs_occ = obs["manual"].get("count")
                if last_obs_occ is None:
                    last_obs_occ = z["capacity"] * 0.5
                all_last_obs_errors.append(abs(last_obs_occ - true_occ))

                # 3. Baseline 2: Prior only
                prior_val = obs.get("prior", {}).get("mean", z["capacity"] * 0.5)
                all_prior_errors.append(abs(prior_val - true_occ))

                # 4. Check coverage of 80% range [free_low, free_high]
                is_covered = (update.free_low <= true_free <= update.free_high)
                if is_covered:
                    covered_count += 1
                total_eval_points += 1

                # Bin by age
                age_m = update.age_seconds / 60.0
                width = update.free_high - update.free_low
                age_to_widths.append((age_m, width))

                if age_m < 5.0:
                    b_key = "<5min"
                elif age_m < 15.0:
                    b_key = "5-15min"
                elif age_m < 30.0:
                    b_key = "15-30min"
                else:
                    b_key = ">30min"

                age_buckets[b_key]["total"] += 1
                if is_covered:
                    age_buckets[b_key]["covered"] += 1
                age_buckets[b_key]["widths"].append(width)

    # Aggregations
    fusion_mae = float(np.mean(all_fusion_errors))
    last_obs_mae = float(np.mean(all_last_obs_errors))
    prior_mae = float(np.mean(all_prior_errors))
    overall_coverage = float(covered_count / total_eval_points) if total_eval_points > 0 else 0.0

    bucket_results = {}
    for k, v in age_buckets.items():
        cov = (v["covered"] / v["total"]) if v["total"] > 0 else 0.0
        avg_w = float(np.mean(v["widths"])) if v["widths"] else 0.0
        bucket_results[k] = {
            "sample_count": v["total"],
            "coverage_rate": round(cov, 4),
            "mean_interval_width": round(avg_w, 2)
        }

    # Plot 1: Interval Width vs Data Age
    plt.figure(figsize=(7, 4.5))
    age_to_widths_sorted = sorted(age_to_widths, key=lambda x: x[0])
    sample_sub = age_to_widths_sorted[::max(1, len(age_to_widths_sorted) // 600)]
    plt.scatter([x[0] for x in sample_sub], [x[1] for x in sample_sub], alpha=0.35, color="#1f77b4", s=18)
    plt.title("80% Interval Width Expansion vs. Observation Age", fontsize=11, fontweight="bold")
    plt.xlabel("Observation Age (Minutes)")
    plt.ylabel("80% Interval Width (Free Slots)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_width_path = output_dir / "interval_width_vs_age.png"
    plt.savefig(plot_width_path, dpi=200)
    plt.close()

    # Plot 2: Coverage by Age Bucket
    plt.figure(figsize=(6, 4))
    b_names = list(bucket_results.keys())
    covs = [bucket_results[k]["coverage_rate"] * 100 for k in b_names]
    plt.bar(b_names, covs, color="#2ca02c", alpha=0.85, edgecolor="black")
    plt.axhline(80.0, color="red", linestyle="--", label="Target Nominal Coverage (80%)")
    plt.ylabel("Empirical Coverage (%)")
    plt.xlabel("Observation Age Bucket")
    plt.title("Empirical Coverage across Observation Age", fontsize=11, fontweight="bold")
    plt.ylim([0, 100])
    plt.legend(loc="lower right")
    plt.tight_layout()
    plot_cov_path = output_dir / "empirical_coverage_by_age.png"
    plt.savefig(plot_cov_path, dpi=200)
    plt.close()

    results = {
        "evaluation_mode": "held_out_seeds_synthetic",
        "held_out_seeds": HELD_OUT_SEEDS,
        "total_eval_points": total_eval_points,
        "metrics": {
            "fusion_mae": round(fusion_mae, 3),
            "baseline_last_obs_mae": round(last_obs_mae, 3),
            "baseline_prior_mae": round(prior_mae, 3),
            "overall_80pct_coverage": round(overall_coverage, 4),
            "coverage_by_age_bucket": bucket_results
        }
    }

    with open(output_dir / "fusion_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n--- FUSION EVALUATION RESULTS ---")
    print(f"Fusion MAE:               {fusion_mae:.3f} slots")
    print(f"Baseline (Last Obs) MAE:  {last_obs_mae:.3f} slots")
    print(f"Baseline (Prior) MAE:     {prior_mae:.3f} slots")
    print(f"Overall 80% Coverage:     {overall_coverage * 100:.2f}% (Target: 80.0%)")
    print("\nCoverage by Age Bucket:")
    for b_k, b_v in bucket_results.items():
        print(f"  {b_k:<10}: Coverage {b_v['coverage_rate']*100:.1f}% | Avg Width {b_v['mean_interval_width']:.1f} slots (N={b_v['sample_count']})")
    print("=" * 65)

    return results


if __name__ == "__main__":
    run_fusion_evaluation()


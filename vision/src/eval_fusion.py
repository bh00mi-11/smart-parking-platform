"""
Fusion calibration evaluation harness on synthetic trajectory streams.
NOTE: This validates the fusion estimation method and uncertainty propagation on synthetic data,
NOT real-world facility accuracy.

Evaluates on HELD-OUT seeds (tuning was conducted strictly on tuning seeds [101, 102]):
1. Occupancy MAE:
   - Fusion estimate vs True Occupancy
   - Baseline 1 (Last observation only) vs True Occupancy
   - Baseline 2 (Historical prior only) vs True Occupancy
   Evaluated under both:
   (a) Non-oracle prior (learned with separate historical noise, shift, unmodeled events)
   (b) Oracle prior (generator curves known in advance)
2. Empirical Coverage of the 80% range using Split-Conformal Calibration per age bucket:
   - Evaluated separately for clean profile and noisy_overhead profile
   - Broken down by data-age buckets: [<5 min, 5-15 min, 15-30 min, >30 min]
   - Reports coverage, mean interval width, sample counts, and MAE
3. Plots:
   - Empirical coverage across observation age
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
TUNING_SEEDS = [101, 102]
HELD_OUT_SEEDS = [201, 202, 203]


def compute_conformal_multipliers(tuning_seeds: List[int]) -> Dict[str, Dict[str, float]]:
    """
    On tuning seeds, for each age bucket (<5, 5-15, 15-30, >30 min) and each noise profile,
    compute the 80% quantile of |residual| / fused_sd and use as that bucket's multiplier.
    """
    zones = [
        {"facility_id": "FAC-TUNING", "zone_id": "ZONE-CLEAN", "capacity": 60, "zone_type": "managed", "profile": "clean"},
        {"facility_id": "FAC-TUNING", "zone_id": "ZONE-NOISY", "capacity": 100, "zone_type": "managed", "profile": "noisy_overhead"},
        {"facility_id": "FAC-TUNING", "zone_id": "ZONE-OPEN", "capacity": 80, "zone_type": "open", "profile": "clean"}
    ]
    
    config = FusionConfig()
    
    ratios = {
        "clean": {"<5min": [], "5-15min": [], "15-30min": [], ">30min": []},
        "noisy_overhead": {"<5min": [], "5-15min": [], "15-30min": [], ">30min": []}
    }
    
    for seed in tuning_seeds:
        for z in zones:
            prof = z["profile"]
            history = simulate_zone_trajectory(
                zone_id=z["zone_id"],
                facility_id=z["facility_id"],
                capacity=z["capacity"],
                zone_type=z["zone_type"],
                noise_profile=prof,
                use_non_oracle_prior=True,
                duration_days=2,
                step_minutes=10,
                camera_interval_minutes=35,
                manual_interval_hours=6,
                seed=seed
            )
            for step in history:
                now_t = step["timestamp"]
                true_occ = step["true_occupancy"]
                true_free = z["capacity"] - true_occ
                obs = step["observations"]
                
                # Estimate with base config to extract standard deviation and residual
                update = estimate_occupancy(z, obs, now=now_t, config=config)
                fused_free = (update.free_low + update.free_high) / 2.0
                resid = abs(fused_free - true_free)
                # Recover approximate fused_sd = width / (2 * z_80)
                width = update.free_high - update.free_low
                fused_sd = max(width / (2.0 * config.z_80), 0.5)
                
                age_m = update.age_seconds / 60.0
                if age_m < 5.0:
                    b_key = "<5min"
                elif age_m < 15.0:
                    b_key = "5-15min"
                elif age_m < 30.0:
                    b_key = "15-30min"
                else:
                    b_key = ">30min"
                    
                ratios[prof][b_key].append(resid / fused_sd)
                
    multipliers = {}
    for prof in ["clean", "noisy_overhead"]:
        multipliers[prof] = {}
        for b_key in ["<5min", "5-15min", "15-30min", ">30min"]:
            vals = ratios[prof][b_key]
            # 80th quantile of |residual| / fused_sd
            q80 = float(np.percentile(vals, 80)) if vals else 1.28
            multipliers[prof][b_key] = round(q80, 3)
            
    return multipliers


def evaluate_stream(
    seeds: List[int],
    use_non_oracle_prior: bool,
    config: FusionConfig
) -> Dict:
    zones = [
        {"facility_id": "FAC-TEST", "zone_id": "ZONE-CLEAN", "capacity": 60, "zone_type": "managed", "profile": "clean"},
        {"facility_id": "FAC-TEST", "zone_id": "ZONE-NOISY", "capacity": 100, "zone_type": "managed", "profile": "noisy_overhead"},
        {"facility_id": "FAC-TEST", "zone_id": "ZONE-OPEN", "capacity": 80, "zone_type": "open", "profile": "clean"}
    ]
    
    all_fusion_errors = []
    all_last_obs_errors = []
    all_prior_errors = []
    
    covered_count = 0
    total_eval_points = 0
    
    # Track per profile and per age bucket
    profile_results = {
        "clean": {
            "<5min": {"total": 0, "covered": 0, "widths": [], "errors": [], "last_obs_err": [], "prior_err": []},
            "5-15min": {"total": 0, "covered": 0, "widths": [], "errors": [], "last_obs_err": [], "prior_err": []},
            "15-30min": {"total": 0, "covered": 0, "widths": [], "errors": [], "last_obs_err": [], "prior_err": []},
            ">30min": {"total": 0, "covered": 0, "widths": [], "errors": [], "last_obs_err": [], "prior_err": []}
        },
        "noisy_overhead": {
            "<5min": {"total": 0, "covered": 0, "widths": [], "errors": [], "last_obs_err": [], "prior_err": []},
            "5-15min": {"total": 0, "covered": 0, "widths": [], "errors": [], "last_obs_err": [], "prior_err": []},
            "15-30min": {"total": 0, "covered": 0, "widths": [], "errors": [], "last_obs_err": [], "prior_err": []},
            ">30min": {"total": 0, "covered": 0, "widths": [], "errors": [], "last_obs_err": [], "prior_err": []}
        }
    }
    
    age_to_widths = []

    for seed in seeds:
        for z in zones:
            prof = z["profile"]
            history = simulate_zone_trajectory(
                zone_id=z["zone_id"],
                facility_id=z["facility_id"],
                capacity=z["capacity"],
                zone_type=z["zone_type"],
                noise_profile=prof,
                use_non_oracle_prior=use_non_oracle_prior,
                duration_days=2,
                step_minutes=10,
                camera_interval_minutes=35,
                manual_interval_hours=6,
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
                last_err = abs(last_obs_occ - true_occ)
                all_last_obs_errors.append(last_err)

                # 3. Baseline 2: Prior only
                prior_val = obs.get("prior", {}).get("mean", z["capacity"] * 0.5)
                prior_err = abs(prior_val - true_occ)
                all_prior_errors.append(prior_err)

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

                b_dict = profile_results[prof][b_key]
                b_dict["total"] += 1
                if is_covered:
                    b_dict["covered"] += 1
                b_dict["widths"].append(width)
                b_dict["errors"].append(fusion_err)
                b_dict["last_obs_err"].append(last_err)
                b_dict["prior_err"].append(prior_err)

    # Compile summaries
    fusion_mae = float(np.mean(all_fusion_errors))
    last_obs_mae = float(np.mean(all_last_obs_errors))
    prior_mae = float(np.mean(all_prior_errors))
    overall_coverage = float(covered_count / total_eval_points) if total_eval_points > 0 else 0.0

    breakdown = {}
    for prof, b_dict in profile_results.items():
        breakdown[prof] = {}
        for b_key, stat in b_dict.items():
            tot = stat["total"]
            cov = (stat["covered"] / tot) if tot > 0 else 0.0
            avg_w = float(np.mean(stat["widths"])) if stat["widths"] else 0.0
            f_mae = float(np.mean(stat["errors"])) if stat["errors"] else 0.0
            l_mae = float(np.mean(stat["last_obs_err"])) if stat["last_obs_err"] else 0.0
            p_mae = float(np.mean(stat["prior_err"])) if stat["prior_err"] else 0.0
            breakdown[prof][b_key] = {
                "sample_count": tot,
                "coverage_rate": round(cov, 4),
                "mean_interval_width": round(avg_w, 2),
                "fusion_mae": round(f_mae, 3),
                "last_obs_mae": round(l_mae, 3),
                "prior_mae": round(p_mae, 3)
            }

    return {
        "fusion_mae": round(fusion_mae, 3),
        "last_obs_mae": round(last_obs_mae, 3),
        "prior_mae": round(prior_mae, 3),
        "overall_coverage": round(overall_coverage, 4),
        "breakdown": breakdown,
        "age_to_widths": age_to_widths
    }


def run_fusion_evaluation(
    output_dir: Union[str, Path] = "vision/outputs"
) -> Dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SPLIT-CONFORMAL FUSION CALIBRATION & EVALUATION")
    print("Tuning seeds:     ", TUNING_SEEDS)
    print("Held-out seeds:   ", HELD_OUT_SEEDS)
    print("=" * 70)

    # Step 1: Tune conformal multipliers strictly on tuning seeds
    print("\n[Step 1] Computing split-conformal multipliers on tuning seeds [101, 102]...")
    conformal_mults = compute_conformal_multipliers(TUNING_SEEDS)
    print("Computed Conformal Multipliers (80% quantiles):")
    for prof, b_dict in conformal_mults.items():
        print(f"  Profile '{prof}':")
        for b_k, m_val in b_dict.items():
            print(f"    {b_k:<10}: {m_val:.3f}")

    config = FusionConfig(conformal_multipliers=conformal_mults)

    # Step 2: Evaluate on HELD-OUT seeds under NON-ORACLE PRIOR
    print("\n[Step 2] Evaluating on HELD-OUT seeds [201, 202, 203] under Non-Oracle Prior...")
    eval_non_oracle = evaluate_stream(HELD_OUT_SEEDS, use_non_oracle_prior=True, config=config)

    # Step 3: Evaluate on HELD-OUT seeds under ORACLE PRIOR
    print("\n[Step 3] Evaluating on HELD-OUT seeds [201, 202, 203] under Oracle Prior...")
    eval_oracle = evaluate_stream(HELD_OUT_SEEDS, use_non_oracle_prior=False, config=config)

    # Plots
    # Plot 1: Interval Width vs Data Age
    plt.figure(figsize=(7, 4.5))
    age_to_widths_sorted = sorted(eval_non_oracle["age_to_widths"], key=lambda x: x[0])
    sample_sub = age_to_widths_sorted[::max(1, len(age_to_widths_sorted) // 600)]
    plt.scatter([x[0] for x in sample_sub], [x[1] for x in sample_sub], alpha=0.35, color="#1f77b4", s=18)
    plt.title("80% Interval Width Expansion vs. Observation Age (Held-Out)", fontsize=11, fontweight="bold")
    plt.xlabel("Observation Age (Minutes)")
    plt.ylabel("80% Interval Width (Free Slots)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_width_path = output_dir / "interval_width_vs_age.png"
    plt.savefig(plot_width_path, dpi=200)
    plt.close()

    # Plot 2: Coverage by Age Bucket (for clean & noisy profiles)
    plt.figure(figsize=(8, 4.5))
    buckets = ["<5min", "5-15min", "15-30min", ">30min"]
    x = np.arange(len(buckets))
    w = 0.35
    cov_clean = [eval_non_oracle["breakdown"]["clean"][b]["coverage_rate"] * 100 for b in buckets]
    cov_noisy = [eval_non_oracle["breakdown"]["noisy_overhead"][b]["coverage_rate"] * 100 for b in buckets]
    plt.bar(x - w/2, cov_clean, width=w, label="Clean Camera (UFPR)", color="#2ca02c", alpha=0.85, edgecolor="black")
    plt.bar(x + w/2, cov_noisy, width=w, label="Noisy Overhead Camera (PUCPR)", color="#d62728", alpha=0.85, edgecolor="black")
    plt.axhline(80.0, color="blue", linestyle="--", label="Target 80% Range")
    plt.axhline(75.0, color="gray", linestyle=":", alpha=0.7)
    plt.axhline(85.0, color="gray", linestyle=":", alpha=0.7)
    plt.ylabel("Empirical Coverage (%)")
    plt.xlabel("Observation Age Bucket")
    plt.title("Split-Conformal Empirical Coverage by Age Bucket (Held-Out Seeds)", fontsize=11, fontweight="bold")
    plt.xticks(x, buckets)
    plt.ylim([0, 100])
    plt.legend(loc="lower right")
    plt.tight_layout()
    plot_cov_path = output_dir / "empirical_coverage_by_age.png"
    plt.savefig(plot_cov_path, dpi=200)
    plt.close()

    results = {
        "evaluation_mode": "held_out_seeds_split_conformal",
        "tuning_seeds": TUNING_SEEDS,
        "held_out_seeds": HELD_OUT_SEEDS,
        "conformal_multipliers": conformal_mults,
        "non_oracle_prior_results": {
            "fusion_mae": eval_non_oracle["fusion_mae"],
            "baseline_last_obs_mae": eval_non_oracle["last_obs_mae"],
            "baseline_prior_mae": eval_non_oracle["prior_mae"],
            "overall_80pct_coverage": eval_non_oracle["overall_coverage"],
            "breakdown": eval_non_oracle["breakdown"]
        },
        "oracle_prior_results": {
            "fusion_mae": eval_oracle["fusion_mae"],
            "baseline_last_obs_mae": eval_oracle["last_obs_mae"],
            "baseline_prior_mae": eval_oracle["prior_mae"],
            "overall_80pct_coverage": eval_oracle["overall_coverage"],
            "breakdown": eval_oracle["breakdown"]
        }
    }

    metrics_path = output_dir / "fusion_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL HELD-OUT RESULTS SUMMARY")
    print("=" * 70)
    print("\n[1] NON-ORACLE PRIOR (Realistic: learned with history noise & unmodeled surges)")
    print(f"  Fusion MAE:               {eval_non_oracle['fusion_mae']:.3f} slots")
    print(f"  Baseline (Last Obs) MAE:  {eval_non_oracle['last_obs_mae']:.3f} slots")
    print(f"  Baseline (Prior Only) MAE:{eval_non_oracle['prior_mae']:.3f} slots")
    print(f"  Overall 80% Coverage:     {eval_non_oracle['overall_coverage']*100:.2f}%")
    print("\n  Clean Profile Breakdown:")
    for b in buckets:
        st = eval_non_oracle["breakdown"]["clean"][b]
        print(f"    {b:<10}: Coverage {st['coverage_rate']*100:.1f}% | Width {st['mean_interval_width']:.1f} slots | MAE {st['fusion_mae']:.2f} (N={st['sample_count']})")
    print("\n  Noisy Overhead Profile Breakdown:")
    for b in buckets:
        st = eval_non_oracle["breakdown"]["noisy_overhead"][b]
        print(f"    {b:<10}: Coverage {st['coverage_rate']*100:.1f}% | Width {st['mean_interval_width']:.1f} slots | MAE {st['fusion_mae']:.2f} (N={st['sample_count']})")

    print("\n[2] ORACLE PRIOR (Generator curves known beforehand)")
    print(f"  Fusion MAE:               {eval_oracle['fusion_mae']:.3f} slots")
    print(f"  Baseline (Last Obs) MAE:  {eval_oracle['last_obs_mae']:.3f} slots")
    print(f"  Baseline (Prior Only) MAE:{eval_oracle['prior_mae']:.3f} slots")
    print(f"  Overall 80% Coverage:     {eval_oracle['overall_coverage']*100:.2f}%")

    print("\nCheck 75%-85% coverage window on held-out seeds:")
    outside_window = []
    for prof in ["clean", "noisy_overhead"]:
        for b in buckets:
            cov = eval_non_oracle["breakdown"][prof][b]["coverage_rate"] * 100
            if cov < 75.0 or cov > 85.0:
                outside_window.append((prof, b, cov))
    if outside_window:
        print("  NOTICE: Buckets outside 75%-85% window:")
        for prof, b, cov in outside_window:
            print(f"    - {prof} {b}: {cov:.1f}%")
    else:
        print("  ALL buckets fall squarely within the 75%-85% target window!")

    print("=" * 70)
    return results


if __name__ == "__main__":
    run_fusion_evaluation()

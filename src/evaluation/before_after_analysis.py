"""Stage 7 — Non-LLM Learning Improvement & Before/After Intervention Analysis.

Computes reproducible statistical comparisons between classroom time intervals
(e.g., baseline instruction vs. post-intervention collaborative session),
evaluating:
- Attention score distributions (mean, std, median)
- High, Moderate, and Low attention proportions
- Valid observation counts, coverage, and missingness rates
- Observable behavioral shifts (Board, Desk/Reading, Slumped, Turned Away, Restless)
- Effect size (Cohen's d) and statistical significance (Welch's t-test / Mann-Whitney U)

CRITICAL RESEARCH CONSTRAINT:
Explicitly separates observable physical behavioral changes from claims of cognitive
learning mastery. Changes in observable attention markers are documented as engagement
shifts, not direct proof of internal cognitive learning outcomes.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats


EPISTEMIC_DISCLAIMER = (
    "SCIENTIFIC INTEGRITY NOTICE: Observable attention scores quantify physical behavioral "
    "markers (head orientation, eye gaze proxy, torso posture, and motor dynamics). "
    "Shifts in attention between intervals indicate changes in observable classroom engagement. "
    "They do not constitute direct measurement of internal cognitive processing or academic learning gains."
)


@dataclass
class IntervalMetrics:
    """Metrics container for a discrete temporal interval."""
    interval_name: str
    start_sec: float
    end_sec: float
    duration_sec: float
    total_frames: int
    valid_observations: int
    missing_observations: int
    missingness_rate: float
    mean_attention: float
    std_attention: float
    median_attention: float
    high_count: int
    high_percentage: float
    moderate_count: int
    moderate_percentage: float
    low_count: int
    low_percentage: float
    uncertain_count: int
    uncertain_percentage: float
    behavior_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BeforeAfterComparisonResult:
    """Statistical before-and-after comparison between two classroom intervals."""
    before_interval: IntervalMetrics
    after_interval: IntervalMetrics
    delta_mean_attention: float
    delta_median_attention: float
    delta_high_percentage: float
    delta_low_percentage: float
    cohens_d: Optional[float]
    effect_size_interpretation: str
    t_statistic: Optional[float]
    p_value: Optional[float]
    is_statistically_significant: bool
    behavioral_shifts: Dict[str, float] = field(default_factory=dict)
    summary_verdict: str = ""
    epistemic_disclaimer: str = EPISTEMIC_DISCLAIMER
    has_comparison: bool = True

    @property
    def pre_interval(self) -> IntervalMetrics:
        return self.before_interval

    @property
    def post_interval(self) -> IntervalMetrics:
        return self.after_interval

    @property
    def statistically_significant(self) -> bool:
        return self.is_statistically_significant

    @property
    def effect_size_label(self) -> str:
        return self.effect_size_interpretation

    @property
    def scientific_disclaimer(self) -> str:
        return self.epistemic_disclaimer

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["has_comparison"] = self.has_comparison
        d["pre_interval"] = self.before_interval.to_dict()
        d["post_interval"] = self.after_interval.to_dict()
        d["statistically_significant"] = self.is_statistically_significant
        d["effect_size_label"] = self.effect_size_interpretation
        d["scientific_disclaimer"] = self.epistemic_disclaimer

        # Tabular behavioral shifts list for clean UI and report rendering
        shifts_list = []
        b_total = max(1, self.before_interval.valid_observations)
        a_total = max(1, self.after_interval.valid_observations)
        all_behaviors = sorted(set(list(self.before_interval.behavior_distribution.keys()) + list(self.after_interval.behavior_distribution.keys())))
        for b in all_behaviors:
            b_cnt = self.before_interval.behavior_distribution.get(b, 0)
            a_cnt = self.after_interval.behavior_distribution.get(b, 0)
            b_pct = round((b_cnt / b_total) * 100.0, 1)
            a_pct = round((a_cnt / a_total) * 100.0, 1)
            delta_pct = round(a_pct - b_pct, 1)
            shifts_list.append({
                "behavior": b,
                "pre_count": b_cnt,
                "pre_percentage": b_pct,
                "post_count": a_cnt,
                "post_percentage": a_pct,
                "delta_percentage": delta_pct
            })
        d["behavioral_shifts"] = shifts_list
        return d


def _compute_interval_metrics(
    name: str,
    start_sec: float,
    end_sec: float,
    student_summaries: List[Dict],
    frame_stats: List[Dict]
) -> IntervalMetrics:
    """Extract metrics from student timelines and frame stats within [start_sec, end_sec]."""
    duration = max(0.001, end_sec - start_sec)
    
    # Filter frame stats
    relevant_frames = [
        f for f in frame_stats
        if start_sec <= f.get("timestamp_sec", 0.0) <= end_sec
    ]
    total_frames = len(relevant_frames)
    
    # Gather observations from student timelines
    scores = []
    categories = []
    behaviors = []
    missing_count = 0

    for s in student_summaries:
        timeline = s.get("timeline", [])
        for obs in timeline:
            ts = obs.get("timestamp_sec", 0.0)
            if start_sec <= ts <= end_sec:
                score = obs.get("score")
                cat = obs.get("category", "Uncertain")
                beh = obs.get("behavior", "Board")
                
                if cat == "Uncertain":
                    missing_count += 1
                
                if score is not None:
                    scores.append(float(score))
                    categories.append(cat)
                    behaviors.append(beh)

    valid_obs = len(scores)
    total_obs = valid_obs + missing_count
    missingness_rate = round(float(missing_count / max(1, total_obs)), 4)

    if scores:
        mean_s = round(float(np.mean(scores)), 2)
        std_s = round(float(np.std(scores, ddof=1) if len(scores) > 1 else 0.0), 2)
        med_s = round(float(np.median(scores)), 2)
    else:
        mean_s, std_s, med_s = 0.0, 0.0, 0.0

    high_c = sum(1 for c in categories if c in ["High", "Attentive"])
    mod_c = sum(1 for c in categories if c == "Moderate")
    low_c = sum(1 for c in categories if c in ["Low", "Inattentive"])
    unc_c = missing_count

    denom = max(1, valid_obs)
    high_pct = round(100.0 * high_c / denom, 2)
    mod_pct = round(100.0 * mod_c / denom, 2)
    low_pct = round(100.0 * low_c / denom, 2)
    unc_pct = round(100.0 * unc_c / max(1, total_obs), 2)

    beh_dist = {}
    for b in set(behaviors):
        beh_dist[b] = behaviors.count(b)

    return IntervalMetrics(
        interval_name=name,
        start_sec=round(start_sec, 2),
        end_sec=round(end_sec, 2),
        duration_sec=round(duration, 2),
        total_frames=total_frames,
        valid_observations=valid_obs,
        missing_observations=missing_count,
        missingness_rate=missingness_rate,
        mean_attention=mean_s,
        std_attention=std_attention_score if (std_attention_score := std_s) is not None else 0.0,
        median_attention=med_s,
        high_count=high_c,
        high_percentage=high_pct,
        moderate_count=mod_c,
        moderate_percentage=mod_pct,
        low_count=low_c,
        low_percentage=low_pct,
        uncertain_count=unc_c,
        uncertain_percentage=unc_pct,
        behavior_distribution=beh_dist
    )


def compare_attention_intervals(
    student_summaries: List[Dict],
    frame_stats: List[Dict],
    split_timestamp_sec: Optional[float] = None,
    before_range: Optional[Tuple[float, float]] = None,
    after_range: Optional[Tuple[float, float]] = None,
    alpha: float = 0.05
) -> BeforeAfterComparisonResult:
    """
    Execute statistical comparison between 'Before' and 'After' intervals.
    
    Can be specified either via:
    1. A single split_timestamp_sec (interval 1: [0, split], interval 2: [split, max_t])
    2. Explicit before_range=(start1, end1) and after_range=(start2, end2)
    """
    # Determine bounds
    all_ts = []
    for s in student_summaries:
        for obs in s.get("timeline", []):
            all_ts.append(obs.get("timestamp_sec", 0.0))
    for f in frame_stats:
        all_ts.append(f.get("timestamp_sec", 0.0))
    
    min_t = min(all_ts) if all_ts else 0.0
    max_t = max(all_ts) if all_ts else 10.0

    if before_range is not None and after_range is not None:
        b_start, b_end = before_range
        a_start, a_end = after_range
    elif split_timestamp_sec is not None:
        b_start, b_end = min_t, split_timestamp_sec
        a_start, a_end = split_timestamp_sec, max_t
    else:
        # Default to 50/50 temporal split
        mid_t = (min_t + max_t) / 2.0
        b_start, b_end = min_t, mid_t
        a_start, a_end = mid_t, max_t

    before_metrics = _compute_interval_metrics("Baseline (Before)", b_start, b_end, student_summaries, frame_stats)
    after_metrics = _compute_interval_metrics("Post-Intervention (After)", a_start, a_end, student_summaries, frame_stats)

    delta_mean = round(after_metrics.mean_attention - before_metrics.mean_attention, 2)
    delta_med = round(after_metrics.median_attention - before_metrics.median_attention, 2)
    delta_high = round(after_metrics.high_percentage - before_metrics.high_percentage, 2)
    delta_low = round(after_metrics.low_percentage - before_metrics.low_percentage, 2)

    # Gather raw score arrays for statistical testing
    b_scores = [
        obs["score"] for s in student_summaries for obs in s.get("timeline", [])
        if b_start <= obs.get("timestamp_sec", 0.0) <= b_end and obs.get("score") is not None
    ]
    a_scores = [
        obs["score"] for s in student_summaries for obs in s.get("timeline", [])
        if a_start <= obs.get("timestamp_sec", 0.0) <= a_end and obs.get("score") is not None
    ]

    # Calculate Cohen's d effect size
    n1, n2 = len(b_scores), len(a_scores)
    has_comp = bool(n1 > 1 and n2 > 1)

    if has_comp:
        s1 = np.std(b_scores, ddof=1)
        s2 = np.std(a_scores, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / max(1, n1 + n2 - 2))
        cohens_d = round(float((after_metrics.mean_attention - before_metrics.mean_attention) / max(1e-6, pooled_std)), 3)
        
        # Welch's 2-sample t-test (unequal variances assumed)
        t_stat, p_val = stats.ttest_ind(a_scores, b_scores, equal_var=False)
        t_stat = round(float(t_stat), 4)
        p_val = round(float(p_val), 6)
        is_sig = bool(p_val < alpha)

        # Interpret effect size
        abs_d = abs(cohens_d)
        if abs_d < 0.20:
            effect_interp = "Negligible"
        elif abs_d < 0.50:
            effect_interp = "Small"
        elif abs_d < 0.80:
            effect_interp = "Medium"
        else:
            effect_interp = "Large"

        direction = "increase" if delta_mean > 0 else "decrease"
        sig_text = "statistically significant" if is_sig else "not statistically significant"
        verdict = (
            f"Observable attention exhibited a {abs(delta_mean):.1f}% {direction} "
            f"(Cohen's d = {cohens_d:+.2f}, {effect_interp} effect, p = {p_val if p_val is not None else 'N/A'}, {sig_text}). "
            f"High-attention proportion changed by {delta_high:+.1f}%, while Low-attention changed by {delta_low:+.1f}%."
        )
        disclaimer = EPISTEMIC_DISCLAIMER
    else:
        cohens_d = None
        t_stat, p_val = None, None
        is_sig = False
        effect_interp = "Insufficient Data"
        verdict = "Insufficient observation data to compute statistical interval comparison."
        disclaimer = EPISTEMIC_DISCLAIMER + " Insufficient observation data for statistical hypothesis testing."

    # Behavioral shifts (% difference for each observable behavior)
    all_behaviors = set(list(before_metrics.behavior_distribution.keys()) + list(after_metrics.behavior_distribution.keys()))
    b_total = max(1, before_metrics.valid_observations)
    a_total = max(1, after_metrics.valid_observations)
    shifts = {}
    for beh in all_behaviors:
        b_rate = before_metrics.behavior_distribution.get(beh, 0) / b_total
        a_rate = after_metrics.behavior_distribution.get(beh, 0) / a_total
        shifts[beh] = round((a_rate - b_rate) * 100.0, 2)

    return BeforeAfterComparisonResult(
        before_interval=before_metrics,
        after_interval=after_metrics,
        delta_mean_attention=delta_mean,
        delta_median_attention=delta_med,
        delta_high_percentage=delta_high,
        delta_low_percentage=delta_low,
        cohens_d=cohens_d,
        effect_size_interpretation=effect_interp,
        t_statistic=t_stat,
        p_value=p_val,
        is_statistically_significant=is_sig,
        behavioral_shifts=shifts,
        summary_verdict=verdict,
        epistemic_disclaimer=disclaimer,
        has_comparison=has_comp
    )

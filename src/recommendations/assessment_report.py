"""Stage 6 — Analysis and Intervention: System Assessment Report Generator.

Aggregates session analytics, student tracking statistics, attention score distributions,
attention drop events, and LLM intervention outputs into a structured System Assessment Report.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.attention_drop.detector import AttentionDropEvent
from src.recommendations.recommendation_engine import Stage6InterventionOutput


@dataclass
class SessionSummaryStats:
    session_id: str
    total_duration_sec: float
    total_students_tracked: int
    mean_attention_score: float
    attentive_ratio: float      # % frames score >= 65
    inattentive_ratio: float    # % frames score <= 40
    moderate_ratio: float       # % frames score 40-65
    total_drop_events: int


class SystemAssessmentReporter:
    """Generates structured session-level pedagogical assessment reports."""

    def __init__(self, output_dir: Optional[Path] = None):
        project_root = Path(__file__).resolve().parent.parent.parent
        self.output_dir = output_dir or (project_root / "reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        session_id: str,
        student_scores_df: pd.DataFrame,
        drop_events: List[AttentionDropEvent],
        interventions: List[Stage6InterventionOutput]
    ) -> Dict:
        """
        Produce comprehensive System Assessment Report in JSON and Markdown formats.
        """
        total_dur = float(student_scores_df["timestamp_sec"].max()) if not student_scores_df.empty else 0.0
        unique_students = int(student_scores_df["track_id"].nunique()) if not student_scores_df.empty else 0
        mean_score = float(student_scores_df["score"].mean()) if not student_scores_df.empty else 50.0

        scores = student_scores_df["score"].values if not student_scores_df.empty else np.array([50.0])
        attentive_pct = float(np.mean(scores >= 65.0) * 100.0)
        inattentive_pct = float(np.mean(scores <= 40.0) * 100.0)
        moderate_pct = float(100.0 - (attentive_pct + inattentive_pct))

        stats = SessionSummaryStats(
            session_id=session_id,
            total_duration_sec=round(total_dur, 2),
            total_students_tracked=unique_students,
            mean_attention_score=round(mean_score, 1),
            attentive_ratio=round(attentive_pct, 1),
            inattentive_ratio=round(inattentive_pct, 1),
            moderate_ratio=round(moderate_pct, 1),
            total_drop_events=len(drop_events)
        )

        report_doc = {
            "session_id": session_id,
            "session_summary": asdict(stats),
            "drop_events": [
                {
                    "event_id": ev.event_id,
                    "track_id": ev.track_id,
                    "timestamp_sec": ev.end_time_sec,
                    "drop_magnitude": ev.drop_magnitude,
                    "evidence": ev.evidence,
                    "confidence": ev.confidence
                }
                for ev in drop_events
            ],
            "interventions": [interv.to_dict() for interv in interventions],
            "ethical_safeguard": (
                "Observable attention metrics reflect visual and postural signals only. "
                "They must not be conflated with cognitive learning outcomes without independent assessment data."
            )
        }

        # Write JSON
        json_path = self.output_dir / "system_assessment_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_doc, f, indent=2)

        # Write Markdown Report
        md_path = self.output_dir / "system_assessment_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# System Assessment Report — Session: {session_id}\n\n")
            f.write(f"**Generated**: Automated Classroom Research Pipeline\n\n")
            f.write("## 1. Session Overview\n\n")
            f.write(f"- **Total Duration**: {stats.total_duration_sec}s\n")
            f.write(f"- **Students Tracked**: {stats.total_students_tracked}\n")
            f.write(f"- **Mean Observable Attention Score**: {stats.mean_attention_score}/100\n")
            f.write(f"- **Attentive Time Ratio (>= 65)**: {stats.attentive_ratio}%\n")
            f.write(f"- **Inattentive Time Ratio (<= 40)**: {stats.inattentive_ratio}%\n")
            f.write(f"- **Total Attention Drop Events**: {stats.total_drop_events}\n\n")

            f.write("## 2. Triggered Interventions & Recommendations\n\n")
            for idx, interv in enumerate(interventions, 1):
                f.write(f"### Intervention #{idx} (Student ID #{interv.target_track_id} at {interv.timestamp_sec:.1f}s)\n\n")
                f.write(f"- **Urgency**: `{interv.urgency_level}`\n")
                f.write(f"- **Explanation**: {interv.explanation}\n")
                f.write(f"- **Teaching Recommendation**: {interv.teaching_recommendation}\n")
                f.write(f"- **Intervention Suggestion**: {interv.intervention_suggestion}\n\n")

            f.write("> [!NOTE]\n")
            f.write(f"> {report_doc['ethical_safeguard']}\n")

        print(f"[ASSESSMENT REPORT] Generated System Assessment Report: {json_path} and {md_path}")
        return report_doc

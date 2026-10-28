"""Stage 6 — Analysis and Intervention: Evidence-Grounded Pedagogical Recommendation Engine.

Generates cautious, evidence-grounded guidance structured strictly into the 3 outputs:
1. Explanation
2. Teaching Recommendation
3. Intervention Suggestion

DOES NOT claim to read internal cognitive mental states directly.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from src.attention_drop.detector import AttentionDropEvent


@dataclass
class Stage6InterventionOutput:
    """Documented output matching Stage 6 diagram outputs exactly."""
    event_id: str
    target_track_id: int
    timestamp_sec: float
    urgency_level: str               # "LOW", "MEDIUM", "HIGH"

    # Exact Output 1: Explanation
    explanation: str

    # Exact Output 2: Teaching Recommendation
    teaching_recommendation: str

    # Exact Output 3: Intervention Suggestion
    intervention_suggestion: str

    # Ethical / Scientific Safeguard
    caution_disclaimer: str = (
        "CAUTION: Analysis is grounded in observable physical markers (head pose, gaze proxy, torso posture, and motion). "
        "It does not directly measure internal cognitive comprehension. "
        "A student looking down may be writing notes or studying quietly."
    )

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "target_track_id": self.target_track_id,
            "timestamp_sec": self.timestamp_sec,
            "urgency_level": self.urgency_level,
            "explanation": self.explanation,
            "teaching_recommendation": self.teaching_recommendation,
            "intervention_suggestion": self.intervention_suggestion,
            "caution_disclaimer": self.caution_disclaimer
        }


class RecommendationEngine:
    """Generates structured, grounded pedagogical guidance from attention drop triggers."""

    def __init__(self, rule_grounding: bool = True):
        self.rule_grounding = rule_grounding
        self.rec_counter = 0

    def generate_recommendation(self, drop_event: AttentionDropEvent) -> Stage6InterventionOutput:
        """
        Generate grounded recommendation matching the 3 Stage 6 diagram outputs.
        """
        self.rec_counter += 1
        ev_lower = drop_event.evidence.lower()

        # Differentiate observational patterns
        if "downward" in ev_lower and "slumped" in ev_lower:
            urgency = "HIGH"
            explanation = (
                f"Student ID #{drop_event.track_id} experienced a sustained score decline from "
                f"{drop_event.baseline_score:.1f} to {drop_event.nadir_score:.1f} (-{drop_event.drop_magnitude:.1f} pts) "
                f"over {drop_event.duration_sec:.1f}s, driven by persistent downward head orientation and upper-body slouching."
            )
            teaching_rec = (
                "Evaluate lesson pacing and cognitive load. The physical slump may indicate cognitive fatigue or lecture fatigue."
            )
            intervention_sugg = (
                "Transition from monologue to a 2-minute 'Think-Pair-Share' active learning exercise, "
                "or introduce a 60-second physical stretch break to re-energize the room."
            )
        elif "oriented away" in ev_lower:
            urgency = "MEDIUM"
            explanation = (
                f"Student ID #{drop_event.track_id} turned head and gaze persistently away from the central instructional board "
                f"for {drop_event.duration_sec:.1f}s (attention drop: -{drop_event.drop_magnitude:.1f} pts)."
            )
            teaching_rec = (
                "Use instructor proximity and visual re-anchoring to re-establish shared classroom focus."
            )
            intervention_sugg = (
                "Move instructor positioning toward that classroom quadrant, and display a high-contrast visual or interactive diagram on screen."
            )
        elif "low motor activity" in ev_lower:
            urgency = "LOW"
            explanation = (
                f"Student ID #{drop_event.track_id} exhibited marked stillness and reduced motor energy over {drop_event.duration_sec:.1f}s. "
                "Observable markers suggest deep individual concentration or passive listening."
            )
            teaching_rec = (
                "Conduct a low-stakes formative pulse check to verify whether stillness represents focus or conceptual impasse."
            )
            intervention_sugg = (
                "Pose a quick digital poll or open comprehension question to the seating zone without putting the individual student on the spot."
            )
        else:
            urgency = "MEDIUM"
            explanation = (
                f"Attention score drop of {drop_event.drop_magnitude:.1f} points detected for student #{drop_event.track_id} "
                f"at timestamp {drop_event.end_time_sec:.1f}s."
            )
            teaching_rec = (
                "Review clarity of the current instruction and allow a short synthesis buffer before moving to the next topic."
            )
            intervention_sugg = (
                "Ask for questions from the class and summarize key takeaways from the previous slide."
            )

        return Stage6InterventionOutput(
            event_id=drop_event.event_id,
            target_track_id=drop_event.track_id,
            timestamp_sec=drop_event.end_time_sec,
            urgency_level=urgency,
            explanation=explanation,
            teaching_recommendation=teaching_rec,
            intervention_suggestion=intervention_sugg
        )

"""Runs Phase Four: Pilot Data Validation."""

import json
from pathlib import Path
from src.ingestion.data_validator import PilotDataValidator

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    pilot_dir = project_root / "data" / "pilot"
    reports_dir = project_root / "reports"

    print("Executing Phase Four: Pilot Data Validation...")
    validator = PilotDataValidator(pilot_dir=pilot_dir, reports_dir=reports_dir)
    report = validator.run_validation()
    print("Validation finished successfully.")

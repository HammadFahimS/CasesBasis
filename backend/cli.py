from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from backend.service import ClinicalCaseSimulator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="clinical-case",
        description="Generate validated clinical case JSON from a diagnosis.",
    )
    parser.add_argument("diagnosis", help="Diagnosis to generate a case for.")
    parser.add_argument(
        "--difficulty",
        choices=("easy", "medium", "hard"),
        default="medium",
        help="Case difficulty. Defaults to medium.",
    )
    parser.add_argument("--model", help="OpenAI model override.")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    simulator = ClinicalCaseSimulator(model=args.model)
    clinical_case = simulator.generate_case(args.diagnosis, difficulty=args.difficulty)
    payload = clinical_case.model_dump_json(indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

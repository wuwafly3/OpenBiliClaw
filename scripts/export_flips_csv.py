"""CSV of first vs second new-contract label flips for JSONL replay."""

import csv
import json
from pathlib import Path

JSONL = Path("E:/otherproject/OpenBiliClaw/data/ml_artifacts/gate_contract_relabel_20260824T030808Z.jsonl")
CSV = Path("E:/otherproject/OpenBiliClaw/data/ml_artifacts/gate_contract_self_consistency_flips.csv")


def main() -> None:
    rows = [json.loads(line) for line in JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]

    with CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "candidate_id",
                "source_platform",
                "source_strategy",
                "first_y",
                "second_y",
                "first_score",
                "second_score",
                "flip",
                "new_profile_digest",
                "new_negative_digest",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["candidate_id"],
                    r["source_platform"],
                    r["source_strategy"],
                    r["first_y"],
                    r["second_y"],
                    r["first_llm_score_raw"],
                    r["second_llm_score_raw"],
                    r["flip"],
                    r["new_profile_digest"],
                    r["new_negative_digest"],
                ]
            )
    print(f"flips csv written: {CSV}")


if __name__ == "__main__":
    main()


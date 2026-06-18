from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_validation.ensemble import validate_document_file_ensemble
from document_validation.validator import ValidationConfig, validate_document_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect blurry, cropped, or non-document image/PDF inputs.",
    )
    parser.add_argument("files", nargs="+", type=Path, help="Image or PDF paths to validate")
    parser.add_argument(
        "--min-document-confidence",
        type=float,
        default=ValidationConfig.min_document_confidence,
        help="Minimum confidence required to classify an image as a document.",
    )
    parser.add_argument(
        "--blur-tenengrad-threshold",
        type=float,
        default=ValidationConfig.blur_tenengrad_threshold,
        help="Tenengrad threshold below which images are considered blurry.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Print one JSON object per input file instead of a JSON array.",
    )
    parser.add_argument(
        "--decision-rows",
        action="store_true",
        help="Print flattened page rows with is_blur/is_cut/is_clear/is_image/Decision.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use weighted max-voting ensemble output instead of the OpenCV-only validator.",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=None,
        choices=["opencv", "opencv_gate", "doctr", "deepdoctection", "deqa_doc"],
        help="Detectors to use with --ensemble. Defaults to all configured detectors.",
    )
    parser.add_argument(
        "--importance",
        nargs="*",
        default=None,
        metavar="DETECTOR=WEIGHT",
        help=(
            "Optional detector importance weights for --ensemble, e.g. "
            "--importance opencv=1.0 doctr=1.25 deqa_doc=1.5"
        ),
    )
    args = parser.parse_args()

    config = ValidationConfig(
        min_document_confidence=args.min_document_confidence,
        blur_tenengrad_threshold=args.blur_tenengrad_threshold,
    )

    results = []
    for file_path in args.files:
        if args.ensemble:
            validation = validate_document_file_ensemble(
                file_path,
                config=config,
                detectors=args.detectors,
                detector_importance=_parse_importance(args.importance),
            )
        else:
            validation = validate_document_file(file_path, config)

        if args.decision_rows:
            rows = [
                {"file": str(file_path), **row}
                for row in validation.to_decision_rows()
            ]
            if args.jsonl:
                for row in rows:
                    print(json.dumps(row, sort_keys=True))
                continue
            results.extend(rows)
            continue

        result = validation.to_dict()
        result["file"] = str(file_path)
        if args.jsonl:
            print(json.dumps(result, sort_keys=True))
        else:
            results.append(result)

    if not args.jsonl:
        print(json.dumps(results, indent=2, sort_keys=True))

    return 0


def _parse_importance(items: list[str] | None) -> dict[str, float] | None:
    if not items:
        return None

    weights: dict[str, float] = {}
    for item in items:
        detector, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"invalid importance value: {item}. Use detector=weight")
        weights[detector] = float(value)
    return weights


if __name__ == "__main__":
    raise SystemExit(main())

"""
Prepare Kera production receipt JSONL data for training.

Input format (one JSON object per line):
    {
        "image_id":     "<uuid>",
        "image_path":   "gs://<bucket>/coverage_images/...",
        "annotated_at": "2026-03-30T14:24:07Z",
        "annotator_id": null,
        "fields": {
            "is_health_receipt": true,
            "total_amount":     15000.0,
            itemized_list:     [
            {
            "item": "item1",
            "unit_price": 5000.0,
            "qty": 2
            },
            ]
            "date":"2025-09-20",
        }
    }

Output format (one JSON object per line):
    {
        "transaction_id": "<uuid>",
        "image_urls":     ["coverage_images/..."],   # GCS path relative to bucket root
        "annotated_at":   "2026-03-30T14:24:07Z",
        "is_health_receipt": true,
        "total_amount":   "15000",  # converted to string for consistency

        "patient_name":   null,
        "provider_info":  null,
        "proof_of_payment": null,
        "date":            "2025-09-20"
    }

The GCS URL  gs://<bucket>/coverage_images/...  is converted to a relative path
by stripping the  gs://<bucket>/  prefix, so downstream code can resolve images
against any mounted bucket root.

Splits produced:
    <output_dir>/train.jsonl       — 90 % of the input train split
    <output_dir>/validation.jsonl  — 10 % of the input train split
    <output_dir>/test.jsonl        — the full input test split (unchanged)
"""

import json
import random
import argparse
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import structlog

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _gcs_url_to_relative_path(gcs_url: str) -> str:
    """
    Strip  gs://<bucket>/  from a GCS URL and return the relative object path.

    Example:
        gs://kera-production.appspot.com/coverage_images/foo.jpg
        -> coverage_images/foo.jpg

    Args:
        gcs_url: A GCS URL starting with gs://.

    Returns:
        The object path relative to the bucket root.  If the URL does not start
        with gs:// it is returned unchanged.
    """
    if not gcs_url.startswith("gs://"):
        return gcs_url
    parsed = urlparse(gcs_url)
    # parsed.path starts with '/', strip the leading slash
    return parsed.path.lstrip("/")


def _convert_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Convert one raw JSONL record to the output task format.

    Args:
        record: Parsed JSON object from the input JSONL file.

    Returns:
        Task dict in the output format.
    """
    fields: dict[str, Any] = record.get("fields") or {}

    # gcs_url is as follow : "gs://kera-production.appspot.com/transaction_proof_images/5bef30a8-..."
    gcs_url: str = record.get("image_path", "")

    # relative_path is as follow:  "transaction_proof_images/5bef30a8-..."
    relative_path = _gcs_url_to_relative_path(gcs_url)

    # total_amount comes as a float (15000.0), convert to clean string
    raw_amount = fields.get("total_amount")
    if isinstance(raw_amount, (int, float)):
        amount_str = str(int(raw_amount)) if raw_amount == int(raw_amount) else str(raw_amount)
    else:
        amount_str = raw_amount  # could be None or already a string

    return {
        "transaction_id": record.get("image_id", ""),
        "image_urls": [relative_path],
        "annotated_at": record.get("annotated_at"),
        "is_health_receipt": fields.get("is_health_receipt", False),
        "total_amount": amount_str,
        "patient_name": fields.get("patient_name"),
        "provider_info": fields.get("provider_info"),
        "proof_of_payment": fields.get("proof_of_payment"),
        "date": fields.get("date"),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Load all records from a JSONL file.

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of parsed JSON objects.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                log.warning(
                    "jsonl_parse_error",
                    path=str(path),
                    line=line_no,
                    error=str(exc),
                )
    return records


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """
    Write a list of dicts to a JSONL file (one JSON object per line).

    Args:
        records: Records to write.
        path:    Destination file path (parent directories are created if needed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def prepare(
    train_jsonl: Path,
    test_jsonl: Path | None,
    output_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Load, convert, split, and save the Kera prescription dataset.

    Args:
        train_jsonl: Input training JSONL file.
        test_jsonl:  Input test JSONL file (optional; skipped when None).
        output_dir:  Directory where train.jsonl, validation.jsonl, and
                     test.jsonl will be written.
        val_ratio:   Fraction of training records to use for validation (0–1).
        seed:        Random seed for the train/validation split.
    """
    # --- Load & convert training data ---
    log.info("loading_train", path=str(train_jsonl))
    raw_train = _load_jsonl(train_jsonl)
    converted_train = [_convert_record(r) for r in raw_train]
    log.info("converted_train", total=len(converted_train))

    # --- Train / validation split ---
    rng = random.Random(seed)
    indices = list(range(len(converted_train)))
    rng.shuffle(indices)

    split_idx = int(len(indices) * (1.0 - val_ratio))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_records = [converted_train[i] for i in train_indices]
    val_records = [converted_train[i] for i in val_indices]

    log.info(
        "split_complete",
        train=len(train_records),
        validation=len(val_records),
        val_ratio=val_ratio,
        seed=seed,
    )

    # --- Write train & validation ---
    _write_jsonl(train_records, output_dir / "train.jsonl")
    log.info("wrote_train", path=str(output_dir / "train.jsonl"), count=len(train_records))

    _write_jsonl(val_records, output_dir / "validation.jsonl")
    log.info("wrote_validation", path=str(output_dir / "validation.jsonl"), count=len(val_records))

    # --- Load, convert & write test data ---
    if test_jsonl is not None:
        log.info("loading_test", path=str(test_jsonl))
        raw_test = _load_jsonl(test_jsonl)
        test_records = [_convert_record(r) for r in raw_test]
        _write_jsonl(test_records, output_dir / "test.jsonl")
        log.info("wrote_test", path=str(output_dir / "test.jsonl"), count=len(test_records))
    else:
        log.info("skipping_test", reason="--test_jsonl not provided")

    log.info("prepare_complete", output_dir=str(output_dir))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Kera production prescription JSONL data for training"
    )
    parser.add_argument(
        "--train_jsonl",
        type=str,
        required=True,
        help="Path to the input training JSONL file",
    )
    parser.add_argument(
        "--test_jsonl",
        type=str,
        default=None,
        help="Path to the input test JSONL file (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="prescription_dataset",
        help="Output directory for train.jsonl, validation.jsonl, and test.jsonl (default: prescription_dataset)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of training data to use for validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/validation split (default: 42)",
    )
    args = parser.parse_args()

    prepare(
        train_jsonl=Path(args.train_jsonl),
        test_jsonl=Path(args.test_jsonl) if args.test_jsonl else None,
        output_dir=Path(args.output_dir),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

"""Validate and apply the target-availability policy."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

POLICY_ID = "known_missing_targets_20260921_v1"
SCHEMA = "diffcloud_known_missing_targets_v1"
EXPECTED_MISSING = {
    "sample_2025012112.h5": (4, 5),
    "sample_2025031012.h5": (4,),
    "sample_2025101112.h5": tuple(range(4, 25)),
    "sample_2025112512.h5": tuple(range(12, 17)),
}
COUNTS = {
    "train": {"dates": 304, "predicted_frames": 7296, "missing_frames": 24, "scored_frames": 7272},
    "val": {"dates": 30, "predicted_frames": 720, "missing_frames": 5, "scored_frames": 715},
    "test": {"dates": 31, "predicted_frames": 744, "missing_frames": 0, "scored_frames": 744},
}
EMPTY_REASON = "all_mask_false_and_all_truth_nonfinite"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Missing or invalid {name} SHA-256")
    return value


def expected_source_names():
    return ["sample_" +
            (datetime(2025, 1, 1) + timedelta(days=i)).strftime("%Y%m%d") + "12.h5"
            for i in range(365)]


def policy_enabled(config):
    value = config.get("dataset", {}).get("known_missing_targets_policy")
    if value is None:
        return False
    if value != POLICY_ID:
        raise ValueError(f"Unknown missing-target policy: {value!r}")
    return True


def _validate_manifest(policy):
    if policy.get("schema") != SCHEMA or policy.get("policy_id") != POLICY_ID:
        raise ValueError("Unknown missing-target manifest schema/policy")
    if policy.get("expected_missing") != {name: list(hours) for name, hours in EXPECTED_MISSING.items()}:
        raise ValueError("Missing-target manifest differs from the expected frame list")
    if policy.get("counts") != COUNTS:
        raise ValueError("Missing-target manifest count contract changed")
    if policy.get("empty_reason") != EMPTY_REASON:
        raise ValueError("Missing-target manifest exception reason changed")
    records = policy.get("source_records", {})
    if set(records) != set(expected_source_names()):
        raise ValueError("Missing-target manifest must bind all 365 daily sources")
    for name, record in records.items():
        if Path(record.get("path", "")).name != name:
            raise ValueError(f"Missing-target source path disagrees with filename: {name}")
        if not isinstance(record.get("size_bytes"), int) or record["size_bytes"] <= 0:
            raise ValueError(f"Invalid source size: {name}")
        if not isinstance(record.get("mtime_ns"), int) or record["mtime_ns"] <= 0:
            raise ValueError(f"Invalid source modification time: {name}")
        _sha(record.get("source_sha256"), f"source {name}")
        _sha(record.get("target_fields_sha256"), f"target fields {name}")
    unsigned = {key: value for key, value in policy.items() if key != "binding_sha256"}
    if policy.get("binding_sha256") != _digest(unsigned):
        raise ValueError("Missing-target manifest content binding changed")
    return policy


def load_policy(config):
    if not policy_enabled(config):
        if config.get("dataset", {}).get("known_missing_targets_manifest"):
            raise ValueError("A missing-target manifest requires its explicit policy ID")
        return None
    path = config.get("dataset", {}).get("known_missing_targets_manifest")
    if not path:
        raise ValueError("Enabled missing-target policy requires an audited manifest")
    encoded = Path(path).read_bytes()
    expected_sha = config.get("dataset", {}).get("known_missing_targets_manifest_sha256")
    if expected_sha is not None and hashlib.sha256(encoded).hexdigest() != expected_sha:
        raise ValueError("Configured missing-target manifest SHA-256 changed")
    return _validate_manifest(json.loads(encoded.decode("utf-8")))


def build_policy_manifest(audit_summary, feature_manifest, sample_dir):
    """Bind a full target-only audit to the independently hashed feature sources."""
    root = Path(sample_dir).resolve()
    audit = audit_summary
    if (audit.get("schema") != "diffcloud_target_availability_audit_v1"
            or audit.get("requested_split") != "all"
            or audit.get("found_files") != 365 or audit.get("expected_files") != 365
            or audit.get("audited_frames") != 8760 or audit.get("empty_frames") != 29
            or audit.get("date_coverage_complete") is not True
            or audit.get("missing_dates") != [] or audit.get("unexpected_dates") != []
            or audit.get("errors") != []):
        raise ValueError("Missing-target policy requires the complete error-free 365-date/8760-frame audit")
    if Path(audit.get("source_directory", "")).resolve() != root:
        raise ValueError("Target audit source directory differs from the configured samples")
    details = audit.get("empty_frame_details", [])
    expected = {(name, hour) for name, hours in EXPECTED_MISSING.items() for hour in hours}
    observed = [(row.get("sample"), row.get("lead_hour")) for row in details]
    if len(observed) != 29 or set(observed) != expected:
        raise ValueError("Audited missing frames differ from the expected frame list")
    for row in details:
        name, hour = row["sample"], row["lead_hour"]
        stamp = name.removeprefix("sample_").removesuffix(".h5")
        init = datetime.strptime(stamp, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        actual = datetime.fromisoformat(str(row.get("target_time_utc", "")).replace("Z", "+00:00"))
        if actual != init + timedelta(hours=hour):
            raise ValueError(f"Missing-target UTC disagrees with sample/lead: {name}, {hour}")
        if row.get("empty_reason") != EMPTY_REASON or any(row.get(key) != 0 for key in (
                "raw_mask_count", "finite_truth_count", "valid_full_grid_count", "valid_east80_count")):
            raise ValueError(f"The approved all-missing target reason was not verified: {name}, {hour}")
    for split, counts in COUNTS.items():
        observed_split = audit.get("by_split", {}).get(split, {})
        if (observed_split.get("audited_frames") != counts["predicted_frames"]
                or observed_split.get("empty_frames") != counts["missing_frames"]):
            raise ValueError(f"Audit split counts disagree: {split}")
    audit_records = audit.get("source_records", {})
    feature_records = feature_manifest.get("files", {})
    names = set(expected_source_names())
    if set(audit_records) != names or set(feature_records) != names:
        raise ValueError("Target audit and feature cache must cover the same complete source year")
    records = {}
    for name in sorted(names):
        source, cached = audit_records[name], feature_records[name]
        if Path(source.get("path", "")).resolve() != (root / name).resolve():
            raise ValueError(f"Target audit source path changed: {name}")
        if (source.get("size_bytes"), source.get("mtime_ns")) != (
                cached.get("source_size"), cached.get("source_mtime_ns")):
            raise ValueError(f"Audit/feature source metadata disagree: {name}")
        full_hash = _sha(cached.get("source_sha256"), f"feature source {name}")
        if source.get("full_source_sha256") not in (None, full_hash):
            raise ValueError(f"Audit/feature source SHA-256 disagree: {name}")
        records[name] = {"path": str((root / name).resolve()),
                         "size_bytes": source["size_bytes"], "mtime_ns": source["mtime_ns"],
                         "target_fields_sha256": _sha(source.get("target_fields_sha256"), name),
                         "source_sha256": full_hash}
    policy = {"schema": SCHEMA, "policy_id": POLICY_ID, "source_directory": str(root),
              "expected_missing": {name: list(hours) for name, hours in EXPECTED_MISSING.items()},
              "empty_reason": EMPTY_REASON, "counts": COUNTS, "source_records": records,
              "audit_summary_sha256": _digest(audit), "feature_manifest_sha256": _digest(feature_manifest)}
    policy["binding_sha256"] = _digest(policy)
    return _validate_manifest(policy)


def missing_leads(policy, sample):
    if policy is None:
        return ()
    name = Path(str(sample)).name
    if not name.endswith(".h5"):
        name += ".h5"
    return tuple(policy["expected_missing"].get(name, ()))


def validate_source_record(policy, path):
    if policy is None:
        return
    path = Path(path)
    record = policy.get("source_records", {}).get(path.name)
    if record is None or path.resolve() != Path(record["path"]).resolve():
        raise ValueError(f"Unbound target-policy source: {path}")
    info = path.stat()
    if (info.st_size, info.st_mtime_ns) != (record["size_bytes"], record["mtime_ns"]):
        raise ValueError(f"Target-policy source metadata changed: {path.name}")

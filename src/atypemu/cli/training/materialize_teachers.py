"""CLI for offline teacher materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atypemu.training import TeacherMaterializationConfig, materialize_teacher_examples


ACTIVE_SUBSET_ENTITY_KEYS = (
    "candidate_free_forced_train_entity_uids",
    "candidate_free_forced_train_val_include_entity_uids",
    "candidate_free_val_include_entity_uids",
    "bioemu_x0_active_subset_protected_entity_uids",
    "candidate_free_replay_entity_uids",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Materialize offline teacher artifacts for staged AtypEmu training.",
    )
    parser.add_argument("--data-root", required=True, help="Repository data root.")
    parser.add_argument(
        "--integrated-root",
        required=True,
        help="Integrated workspace root such as data/integrated.",
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help="Optional teacher-materialization config JSON.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Split to materialize. Repeat for multiple splits.",
    )
    parser.add_argument(
        "--shift-template",
        action="append",
        default=[],
        help="Chemical-shift template mapping in SOURCE=PATH format.",
    )
    parser.add_argument(
        "--shift-format",
        action="append",
        default=[],
        help="Chemical-shift format mapping in SOURCE=FORMAT format.",
    )
    parser.add_argument(
        "--refresh-outputs",
        action="store_true",
        help="Recompute per-example pool, observable, and teacher artifacts.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of teacher examples.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=None,
        help="Entry-level multiprocessing worker count for materialization.",
    )
    parser.add_argument(
        "--entity-uid",
        action="append",
        default=[],
        help="Restrict materialization to one entity UID. Repeat for multiple entities.",
    )
    parser.add_argument(
        "--bmrb-id",
        action="append",
        default=[],
        help="Restrict materialization to one BMRB ID. Repeat for multiple IDs.",
    )
    parser.add_argument(
        "--student-config-json",
        default=None,
        help="Student config used to derive an active-subset materialization filter.",
    )
    parser.add_argument(
        "--active-subset-only",
        action="store_true",
        help=(
            "Materialize only student-config active/forced/replay entity UIDs. "
            "Fails closed if the student config has no explicit entity list."
        ),
    )
    return parser


def _list_payload(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _active_subset_entity_uids(student_config_path: str) -> list[str]:
    payload = json.loads(Path(student_config_path).read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"student config must be a JSON object: {student_config_path}")
    entity_uids: list[str] = []
    seen: set[str] = set()
    for key in ACTIVE_SUBSET_ENTITY_KEYS:
        for entity_uid in _list_payload(payload.get(key)):
            if entity_uid not in seen:
                entity_uids.append(entity_uid)
                seen.add(entity_uid)
    audit = payload.get("experiment_audit")
    if isinstance(audit, dict):
        for entity_uid in _list_payload(audit.get("expected_teacher_entities")):
            if entity_uid not in seen:
                entity_uids.append(entity_uid)
                seen.add(entity_uid)
    if not entity_uids:
        raise SystemExit(
            "--active-subset-only requires explicit entity UIDs in the student config"
        )
    return entity_uids


def main() -> None:
    """Run the teacher-materialization CLI."""
    args = build_parser().parse_args()
    config = (
        TeacherMaterializationConfig.from_json(args.config_json)
        if args.config_json
        else TeacherMaterializationConfig()
    )
    if args.split:
        config.selected_splits = list(args.split)
    if args.max_examples is not None:
        config.max_examples = args.max_examples
    if args.worker_count is not None:
        config.materialization_worker_count = max(1, int(args.worker_count))
    if args.entity_uid:
        config.selected_entity_uids.extend(args.entity_uid)
    if args.bmrb_id:
        config.selected_bmrb_ids.extend(args.bmrb_id)
    if args.active_subset_only:
        if not args.student_config_json:
            raise SystemExit("--active-subset-only requires --student-config-json")
        config.selected_entity_uids.extend(
            _active_subset_entity_uids(args.student_config_json)
        )
    if args.refresh_outputs:
        config.refresh_outputs = True
    for mapping in args.shift_template:
        source, value = mapping.split("=", 1)
        config.chemical_shift_dir_templates[source] = value
    for mapping in args.shift_format:
        source, value = mapping.split("=", 1)
        config.chemical_shift_formats[source] = value

    summary = materialize_teacher_examples(
        data_root=args.data_root,
        integrated_root=args.integrated_root,
        config=config,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

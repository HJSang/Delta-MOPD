"""Local tokenizer/provenance checks; never silently project cross-family IDs."""

import json
from pathlib import Path

from transformers import AutoTokenizer


def validate_vocabulary_maps(vocabularies: dict[str, dict[str, int]], effective_vocab_size: int) -> None:
    reference = None
    for name, vocab in vocabularies.items():
        current = vocab
        if len(current) != effective_vocab_size or set(current.values()) != set(range(effective_vocab_size)):
            raise ValueError(f"{name}: effective vocabulary must be complete, contiguous and one-to-one.")
        if reference is not None and current != reference:
            raise ValueError(
                f"{name}: tokenizer IDs differ; Appendix F projection is not implemented in this adapter."
            )
        reference = current
    if reference is None:
        raise ValueError("Missing tokenizer vocabularies.")


def validate_paper_model_manifest(args) -> None:
    """Manifest records local tokenizer/model paths and pinned source revisions.

    Deployment must separately verify that each URL/LoRA serves these weights.
    This check proves tokenizer compatibility, NOT the provenance of a remote
    engine's in-memory weights. The anchor must be the student initialization.
    """
    manifest_path = getattr(args, "opd_paper_model_manifest", None)
    if not manifest_path:
        raise ValueError("Paper OPD requires --opd-paper-model-manifest (pinned model/tokenizer identities).")
    manifest = json.loads(Path(manifest_path).read_text())
    models = {"student": manifest["student"], "anchor": manifest["anchor"]}
    teacher_names = {entry.split("=", 1)[0] for entry in args.opd_teacher_urls}
    if set(manifest["teachers"]) != teacher_names:
        raise ValueError("Manifest teacher names must match the scoring routes.")
    if args.opd_target_mode == "shiftmopd" and set(manifest["bases"]) != teacher_names:
        raise ValueError("Manifest must identify each selected teacher's exact precursor.")
    if args.opd_target_mode == "shiftmopd":
        route_base_names = {entry.split("=", 1)[0] for entry in args.opd_base_urls or []}
        if route_base_names != teacher_names:
            raise ValueError("Teacher and base scoring route names must match.")
    for kind in ("teachers", "bases"):
        models.update({f"{kind}/{name}": model for name, model in manifest.get(kind, {}).items()})
    if Path(manifest["student"]["path"]).resolve() != Path(args.hf_checkpoint).resolve():
        raise ValueError("Manifest student must be the --hf-checkpoint initialization.")
    for key in ("repo_id", "revision"):
        if manifest["anchor"][key] != manifest["student"][key]:
            raise ValueError("Frozen anchor identity must equal the initial student identity.")
    vocabularies = {}
    for name, model in models.items():
        if not model.get("repo_id") or not model.get("revision") or model["revision"] in ("main", "latest"):
            raise ValueError(f"{name}: record a pinned source revision, not a moving main/latest branch.")
        tokenizer = AutoTokenizer.from_pretrained(model["path"], local_files_only=True, trust_remote_code=False)
        vocabularies[name] = tokenizer.get_vocab()
    validate_vocabulary_maps(vocabularies, args.opd_effective_vocab_size)

"""Merge vendor/_clone_results.json into each registry entry's `audit` field (CLONE_ALL_PROMPT.md).

Does NOT change wave, app_tier, integrated or verified. Adds "permission": "granted by author" to every entry
(the project owner has written permission from every repo author; see docs/permissions/).
Writes docs/model_registry.json and backend/registry/model_registry.json (kept identical).
"""
import json
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
DOCS_REGISTRY = BACKEND.parent / "docs" / "model_registry.json"
API_REGISTRY = BACKEND / "registry" / "model_registry.json"
RESULTS = BACKEND / "vendor" / "_clone_results.json"

AUDIT_KEYS = ["status", "error", "cloned", "commit", "last_commit", "size_mb", "removed_dirs", "skipped_paths", "lfs_pull",
              "weight_files", "frameworks", "license", "external_downloads", "notebooks_only", "has_requirements",
              "file_count", "notebooks", "python_files"]


def main() -> None:
    registry = json.loads(DOCS_REGISTRY.read_text(encoding="utf-8"))
    results = {r["id"]: r for r in json.loads(RESULTS.read_text(encoding="utf-8"))}
    missing = []
    for entry in registry["models"]:
        res = results.get(entry["id"])
        if res is None:
            missing.append(entry["id"])
            continue
        audit = dict(entry.get("audit") or {})
        audit.update({"clone_" + k if k in ("status", "error") else k: res[k] for k in AUDIT_KEYS if k in res})
        audit["weights_in_repo"] = bool(res.get("weight_files"))
        audit["checked"] = res.get("cloned", audit.get("checked"))
        entry["audit"] = audit
        entry["permission"] = "granted by author"
    text = json.dumps(registry, indent=1)
    DOCS_REGISTRY.write_text(text, encoding="utf-8")
    API_REGISTRY.write_text(text, encoding="utf-8")
    print(f"updated {len(registry['models']) - len(missing)} entries; not in results: {missing or 'none'}")


if __name__ == "__main__":
    main()

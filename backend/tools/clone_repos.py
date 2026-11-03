"""Clone all model repos into backend/vendor/ and audit them (docs/CLONE_ALL_PROMPT.md, WORKFLOW.md A1).

The repos are UNTRUSTED. This script only runs `git clone` / `git lfs pull` and reads files as bytes or text.
It never imports, executes or installs anything from them. Git hooks are disabled and LFS smudging is skipped,
except for weight files, which are pulled one by one.

Usage (from backend/):  python tools/clone_repos.py [--only ID ...] [--workers 6]
Writes vendor/_clone_results.json; tools/update_registry_audit.py merges it into the registry.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND.parent
VENDOR = BACKEND / "vendor"
REPOS_TXT = REPO_ROOT / "docs" / "repos.txt"
RESULTS = VENDOR / "_clone_results.json"

MIN_FREE_GB = 25
MAX_REPO_BYTES = 2 * 1024**3
WEIGHT_EXT = {".pkl", ".joblib", ".sav", ".h5", ".hdf5", ".keras", ".pt", ".pth", ".onnx", ".tflite",
              ".safetensors", ".ckpt", ".pb", ".bin", ".model", ".cbm", ".ubj"}
MIN_BIN_BYTES = 100 * 1024  # .bin/.model files smaller than this are rarely weights
DATA_DIRS = {"data", "dataset", "datasets", "images", "image", "imgs", "img", "samples", "sample_images",
             "raw_data", "raw", "test_images", "train_images", "val_images", "dicom", "scans"}
FRAMEWORKS = {
    "sklearn": r"\bsklearn\b|scikit-learn",
    "xgboost": r"\bxgboost\b",
    "lightgbm": r"\blightgbm\b",
    "catboost": r"\bcatboost\b",
    "pytorch": r"\bimport torch\b|\bfrom torch\b|\btorch==|\btorch>=|^torch\s*$",
    "tensorflow": r"\btensorflow\b|\bfrom keras\b|\bimport keras\b",
    "ultralytics": r"\bultralytics\b|\byolov?8\b",
    "onnx": r"\bonnxruntime\b|\bimport onnx\b",
    "monai": r"\bmonai\b",
    "timm": r"\bimport timm\b|\btimm\b",
    "r-lang": r"\blibrary\(",
}
LINK_HOSTS = {"google_drive": r"drive\.google\.com|docs\.google\.com/uc", "huggingface": r"huggingface\.co/",
              "zenodo": r"zenodo\.org/", "kaggle": r"kaggle\.com/", "dropbox": r"dropbox\.com/",
              "github_release": r"github\.com/[^\s)]+/releases", "onedrive": r"1drv\.ms|onedrive\.live\.com",
              "figshare": r"figshare\.com/", "physionet": r"physionet\.org/"}
CODE_EXT = {".py", ".ipynb", ".txt", ".toml", ".cfg", ".yaml", ".yml", ".r", ".R"}


NO_HOOKS = VENDOR / ".no-hooks"  # absolute, outside every repo; "/dev/null" becomes a folder inside the repo on Windows


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 900) -> subprocess.CompletedProcess:
    # GIT_CEILING_DIRECTORIES: git must never walk up from a vendor folder into the Curanex repo itself
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0", "GIT_CEILING_DIRECTORIES": str(VENDOR)}
    if cmd[:1] == ["git"]:
        cmd = ["git", "-c", f"core.hooksPath={NO_HOOKS.as_posix()}", *cmd[1:]]
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout,  # noqa: S603 fixed git argv, no shell
                          encoding="utf-8", errors="replace")


METADATA_ONLY_LIST = BACKEND / "tools" / "metadata_only_repos.txt"


def metadata_only_ids() -> set[str]:
    """Repos kept as Git metadata only (no files) by the owner's decision; one model_id per line, # comments."""
    if not METADATA_ONLY_LIST.exists():
        return set()
    lines = METADATA_ONLY_LIST.read_text(encoding="utf-8").splitlines()
    return {line.split("#")[0].strip() for line in lines if line.split("#")[0].strip()}


def remove_tree(path: Path) -> None:
    """rmtree that also deletes Git's read-only object files on Windows."""
    def make_writable(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if path.exists():
        shutil.rmtree(path, onexc=make_writable)


WINDOWS_BAD = re.compile(r'[<>:"|?*\x00-\x1f]|[ .]$|^(con|prn|aux|nul|com\d|lpt\d)(\..*)?$', re.I)


def checkout_valid_paths(dest: Path) -> list[str] | None:
    """Check out every file whose path can exist on this OS. Paths that can't (e.g. a ':' in a folder name on
    Windows) are excluded with a sparse checkout and returned. None means the checkout failed."""
    names = run(["git", "ls-tree", "-r", "--name-only", "-z", "HEAD"], cwd=dest).stdout.split("\0")
    bad_prefixes = set()
    for name in filter(None, names):
        parts = name.split("/")
        for i, part in enumerate(parts):
            if os.name == "nt" and WINDOWS_BAD.search(part):
                bad_prefixes.add("/".join(parts[: i + 1]))
                break
    if bad_prefixes:
        patterns = ["/*"] + ["!/" + re.sub(r"([*?\[\\!#])", r"\\\1", p) + ("/" if p not in names else "")
                             for p in sorted(bad_prefixes)]
        if run(["git", "sparse-checkout", "set", "--no-cone", *patterns], cwd=dest).returncode != 0:
            return None
    if run(["git", "checkout", "--quiet", "HEAD"], cwd=dest, timeout=1800).returncode != 0:
        return None
    return sorted(bad_prefixes)


def audit_from_tree(dest: Path, hash_weights: bool = True) -> dict:
    """Same audit fields as audit(), read from Git objects (no working tree). Blobs are read as bytes only."""
    def blob(oid: str) -> bytes:
        env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(VENDOR)}
        return subprocess.run(["git", "cat-file", "-p", oid], cwd=dest, env=env,  # noqa: S603,S607 fixed git argv
                              capture_output=True, timeout=300).stdout

    entries = []
    # `ls-tree -l` needs every blob for its size, which downloads the whole repo in a blob-less clone:
    # metadata-only clones list names only (size 0 = unknown)
    long_listing = ["-l"] if hash_weights else []
    for line in run(["git", "ls-tree", "-r", *long_listing, "-z", "HEAD"], cwd=dest).stdout.split("\0"):
        if line:
            meta, path = line.split("\t", 1)
            fields = meta.split()
            if fields[1] == "blob":
                entries.append((path, fields[2], int(fields[3]) if hash_weights else 0))
    weights, frameworks, links = [], set(), {}
    for path, oid, size in entries:
        p = Path(path)
        ext = p.suffix.lower()
        if ext in WEIGHT_EXT and (ext not in {".bin", ".model"} or size >= MIN_BIN_BYTES):
            if not hash_weights:  # metadata-only: don't download weights just to hash them
                weights.append({"path": path, "size": size, "lfs_pointer": None, "sha256": None, "downloaded": False})
                continue
            data = blob(oid)
            pointer = data[:40].startswith(b"version https://git-lfs")
            weights.append({"path": path, "size": size, "lfs_pointer": pointer,
                            "sha256": None if pointer else hashlib.sha256(data).hexdigest()})
        # metadata-only clones fetch each file on demand, so only requirements files are read for frameworks there
        if (hash_weights and ext in CODE_EXT and size < 5_000_000) or p.name.lower().startswith("requirements"):
            text = blob(oid).decode("utf-8", errors="replace")
            frameworks |= {name for name, pat in FRAMEWORKS.items() if re.search(pat, text, re.M)}
        if p.name.lower().startswith("readme") and "/" not in path:
            text = blob(oid).decode("utf-8", errors="replace")
            for host, pat in LINK_HOSTS.items():
                for m in re.finditer(rf"https?://[^\s)\]\"'<>]*(?:{pat})[^\s)\]\"'<>]*", text):
                    links.setdefault(host, set()).add(m.group(0))
    lic = next((oid for path, oid, _ in entries if "/" not in path and re.match(r"(?i)^(licen[cs]e|copying)", path)), None)
    license_name = "none found"
    if lic:
        tmp = dest / ".license_probe"
        tmp.mkdir(exist_ok=True)
        (tmp / "LICENSE").write_bytes(blob(lic))
        license_name = classify_license(tmp)
        remove_tree(tmp)
    return {
        "commit": run(["git", "rev-parse", "HEAD"], cwd=dest).stdout.strip(),
        "last_commit": run(["git", "log", "-1", "--format=%cs"], cwd=dest).stdout.strip(),
        "size_mb": round(sum(s for _, _, s in entries) / 1024**2, 1) if hash_weights else None,
        "weight_files": weights, "frameworks": sorted(frameworks), "license": license_name,
        "external_downloads": [{"host": h, "url": u} for h, us in sorted(links.items()) for u in sorted(us)][:20],
        "notebooks_only": any(p.endswith(".ipynb") for p, _, _ in entries)
                          and not any(p.endswith(".py") and s > 200 for p, _, s in entries),
        "has_requirements": any(Path(p).name.lower().startswith("requirements") for p, _, _ in entries),
        "file_count": len(entries),
        "notebooks": sum(p.endswith(".ipynb") for p, _, _ in entries),
        "python_files": sum(p.endswith(".py") for p, _, _ in entries),
    }


def is_clone_of(dest: Path, url: str) -> bool:
    """True only for a complete checkout whose top level is `dest` and whose origin is `url`."""
    if not (dest / ".git").is_dir():
        return False
    top = run(["git", "rev-parse", "--show-toplevel"], cwd=dest)
    origin = run(["git", "remote", "get-url", "origin"], cwd=dest).stdout.strip()
    head = run(["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=dest)
    return (top.returncode == 0 and Path(top.stdout.strip()).resolve() == dest.resolve()
            and origin.rstrip("/").removesuffix(".git") == url.rstrip("/") and head.returncode == 0)


def parse_repos() -> list[tuple[str, str, str]]:
    out, wave = [], ""
    for line in REPOS_TXT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#"):
            wave = line.lstrip("# ").split(" (")[0]
        elif line:
            url, model_id = line.split()
            out.append((url, model_id, wave))
    return out


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and ".git" not in f.parts)


def is_lfs_pointer(f: Path) -> bool:
    if f.stat().st_size > 1024:
        return False
    return f.read_bytes()[:40].startswith(b"version https://git-lfs")


def sha256(f: Path) -> str:
    h = hashlib.sha256()
    with f.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_weight(f: Path) -> bool:
    ext = f.suffix.lower()
    if ext not in WEIGHT_EXT:
        return False
    return ext not in {".bin", ".model"} or f.stat().st_size >= MIN_BIN_BYTES or is_lfs_pointer(f)


def classify_license(repo: Path) -> str:
    files = [p for p in repo.iterdir() if p.is_file() and re.match(r"(?i)^(licen[cs]e|copying)", p.name)]
    if not files:
        return "none found"
    text = files[0].read_text(encoding="utf-8", errors="replace")[:4000].lower()
    for name, pat in [("MIT", r"\bmit license\b|permission is hereby granted, free of charge"),
                      ("Apache-2.0", r"apache license|apache\.org/licenses/license-2\.0"),
                      ("GPL-3.0", r"gnu general public license.{0,40}version 3"),
                      ("GPL-2.0", r"gnu general public license.{0,40}version 2"), ("AGPL", r"affero"),
                      ("LGPL", r"lesser general public"), ("BSD", r"\bbsd\b|redistribution and use in source"),
                      ("CC", r"creative commons"), ("MPL-2.0", r"mozilla public license"), ("Unlicense", r"unlicense")]:
        if re.search(pat, text, re.S):
            return name
    return "other"


def audit(repo: Path) -> dict:
    files = [f for f in repo.rglob("*") if f.is_file() and ".git" not in f.parts]
    weights, frameworks, links = [], set(), {}
    py = [f for f in files if f.suffix == ".py"]
    nb = [f for f in files if f.suffix == ".ipynb"]
    for f in files:
        rel = f.relative_to(repo).as_posix()
        if is_weight(f):
            pointer = is_lfs_pointer(f)
            weights.append({"path": rel, "size": f.stat().st_size, "lfs_pointer": pointer,
                            "sha256": None if pointer else sha256(f)})
        if f.suffix in CODE_EXT and f.stat().st_size < 5_000_000 or f.name.lower().startswith("requirements"):
            text = f.read_text(encoding="utf-8", errors="replace")
            for name, pat in FRAMEWORKS.items():
                if re.search(pat, text, re.M):
                    frameworks.add(name)
    for readme in [f for f in files if f.name.lower().startswith("readme")][:3]:
        text = readme.read_text(encoding="utf-8", errors="replace")
        for host, pat in LINK_HOSTS.items():
            for m in re.finditer(rf"https?://[^\s)\]\"'<>]*(?:{pat})[^\s)\]\"'<>]*", text):
                links.setdefault(host, set()).add(m.group(0))
    for w in weights:
        ext = Path(w["path"]).suffix.lower()
        if ext == ".onnx":
            frameworks.add("onnx")
    return {
        "weight_files": weights,
        "frameworks": sorted(frameworks),
        "license": classify_license(repo),
        "external_downloads": [{"host": h, "url": u} for h, us in sorted(links.items()) for u in sorted(us)][:20],
        "notebooks_only": bool(nb) and not any(f.stat().st_size > 200 for f in py),
        "has_requirements": any(f.name.lower().startswith("requirements") or f.name in
                                ("environment.yml", "pyproject.toml", "setup.py") for f in files),
        "file_count": len(files),
        "notebooks": len(nb),
        "python_files": len(py),
    }


def trim_data_dirs(repo: Path) -> list[str]:
    """Delete dataset/image folders (keeping weight files) from an oversized repo."""
    removed = []
    for d in sorted(repo.rglob("*"), key=lambda p: len(p.parts)):
        if d.is_dir() and ".git" not in d.parts and d.name.lower() in DATA_DIRS and d.exists():
            if any(is_weight(f) for f in d.rglob("*") if f.is_file()):
                for f in d.rglob("*"):
                    if f.is_file() and not is_weight(f):
                        f.unlink()
            else:
                shutil.rmtree(d, ignore_errors=True)
            removed.append(d.relative_to(repo).as_posix())
    return removed


def clone_one(url: str, model_id: str, wave: str) -> dict:
    dest = VENDOR / model_id
    result = {"id": model_id, "url": url, "wave": wave, "cloned": str(date.today())}
    if model_id in metadata_only_ids():  # owner decision: keep Git metadata, no files, no LFS weights
        partial = run(["git", "config", "remote.origin.partialclonefilter"], cwd=dest).stdout.strip() if dest.exists() else ""
        if not is_clone_of(dest, url) or any(f.name != ".git" for f in dest.iterdir()) or partial != "blob:none":
            remove_tree(dest)  # WORKFLOW.md Part E: names and sizes only, file contents fetched on demand
            if run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", "--quiet",
                    url, str(dest)]).returncode != 0:
                return {**result, "status": "failed", "error": "metadata clone failed"}
        return {**result, "status": "metadata_only", **audit_from_tree(dest, hash_weights=False),
                "error": f"kept as metadata only on purpose (listed in {METADATA_ONLY_LIST.name})"}
    if not is_clone_of(dest, url):  # missing, or a broken/partial clone from an interrupted run
        remove_tree(dest)
        err = ""
        for _ in range(2):  # retry once on network errors
            r = run(["git", "clone", "--depth", "1", "--no-checkout", "--quiet", url, str(dest)])
            if r.returncode == 0:
                break
            err = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["unknown error"]
            remove_tree(dest)
        else:
            return {**result, "status": "failed", "error": err[0][:300]}
        skipped = checkout_valid_paths(dest)
        if skipped is None:  # paths this OS can't hold: record metadata from Git's objects (WORKFLOW.md A1)
            return {**result, "status": "metadata_only", **audit_from_tree(dest),
                    "error": "files not checked out: the repo has paths Windows can't create (e.g. ':' in folder names)"}
        if skipped:
            result["skipped_paths"] = skipped
        if not is_clone_of(dest, url):
            return {**result, "status": "failed", "error": "clone finished but the folder is not a checkout of this repo"}
    elif not any(f.name != ".git" for f in dest.iterdir()):  # kept from an earlier metadata-only run
        return {**result, "status": "metadata_only", **audit_from_tree(dest),
                "error": "files not checked out: the repo has paths Windows can't create (e.g. ':' in folder names)"}
    result["commit"] = run(["git", "rev-parse", "HEAD"], cwd=dest).stdout.strip()
    result["last_commit"] = run(["git", "log", "-1", "--format=%cs"], cwd=dest).stdout.strip()

    # Pull real weight files that are stored as LFS pointers (and nothing else from LFS)
    pointers = [f.relative_to(dest).as_posix() for f in dest.rglob("*")
                if f.is_file() and ".git" not in f.parts and f.suffix.lower() in WEIGHT_EXT and is_lfs_pointer(f)]
    if pointers:
        r = run(["git", "lfs", "pull", "--include", ",".join(pointers)], cwd=dest, timeout=1800)
        result["lfs_pull"] = "ok" if r.returncode == 0 else (r.stderr.strip().splitlines() or ["failed"])[-1][:300]

    stray = dest / "dev" / "null"  # left by older runs that used core.hooksPath=/dev/null
    if stray.is_dir() and not run(["git", "ls-files", "dev/null"], cwd=dest).stdout.strip():
        shutil.rmtree(stray, ignore_errors=True)
        if not any((dest / "dev").iterdir()):
            (dest / "dev").rmdir()

    size = dir_size(dest)
    if size > MAX_REPO_BYTES:
        result["removed_dirs"] = trim_data_dirs(dest)
        size = dir_size(dest)
        if size > MAX_REPO_BYTES:
            result["status"] = "needs_decision"
    result["size_mb"] = round(size / 1024**2, 1)
    result.setdefault("status", "ok")
    result.update(audit(dest))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    free_gb = shutil.disk_usage(BACKEND).free / 1024**3
    print(f"Free disk: {free_gb:.0f} GB")
    if free_gb < MIN_FREE_GB:
        print(f"Less than {MIN_FREE_GB} GB free. Stopping.")
        return 2
    VENDOR.mkdir(exist_ok=True)
    repos = [r for r in parse_repos() if not args.only or r[1] in args.only]
    previous = {r["id"]: r for r in json.loads(RESULTS.read_text())} if RESULTS.exists() else {}

    def task(r):
        try:
            res = clone_one(*r)
        except Exception as e:  # keep going; report it
            res = {"id": r[1], "url": r[0], "wave": r[2], "status": "failed", "error": f"{type(e).__name__}: {e}"[:300]}
        print(f"[{res['status']:>14}] {res['id']} {res.get('size_mb', '')} MB {res.get('error', '')}", flush=True)
        return res

    with ThreadPoolExecutor(args.workers) as pool:
        for res in pool.map(task, repos):
            old = previous.get(res["id"], {})
            if "size_mb" in res and res["size_mb"] is None and old.get("size_mb"):  # metadata-only: keep measured size
                res["size_mb"] = old["size_mb"]
                res["size_note"] = old.get("size_note") or f"measured on the full clone of {old.get('cloned')}"
            previous[res["id"]] = res
    RESULTS.write_text(json.dumps(sorted(previous.values(), key=lambda r: r["id"]), indent=1))
    done = list(previous.values())
    print(f"\nok={sum(r['status'] == 'ok' for r in done)} failed={sum(r['status'] == 'failed' for r in done)} "
          f"needs_decision={sum(r['status'] == 'needs_decision' for r in done)} "
          f"metadata_only={sum(r['status'] == 'metadata_only' for r in done)} "
          f"total={sum(r.get('size_mb', 0) for r in done) / 1024:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
